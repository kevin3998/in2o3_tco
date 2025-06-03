import os
import sqlite3
import requests
import time
import tenacity
from tqdm import tqdm
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from datetime import datetime
import re
import random
from html import unescape
from nltk.stem import PorterStemmer
import json
from fake_useragent import UserAgent
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed  # 新增并发库
import psutil  # 新增系统信息库

# ------------------ 配置参数 ------------------
EMAIL = "chenlintao3998@gmail.com"
SAVE_DIR = "./In2O3-TCO"
DB_PATH = "In2O3-TCO.db"
START_YEAR = 2012
END_YEAR = 2013
ROWS_PER_PAGE = 100
MAX_PAGES = 100
API_CALL_INTERVAL= 30
REQUEST_TIMEOUT = 30

# ------------------ 代理配置 ------------------
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=5000&country=us,gb,de"
]

PROXY_REFRESH_INTERVAL = 1800
PROXY_TEST_URL = "http://www.google.com"
PROXY_TIMEOUT = 2  # 缩短基础超时时间

# ------------------ 动态User-Agent生成器 ------------------
ua = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

def get_random_headers():
    """生成动态请求头"""
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": str(random.randint(0, 1))
    }

# ------------------ 代理池优化 ------------------
proxy_pool = []
proxy_lock = threading.Lock()
last_proxy_refresh = 0


def auto_max_workers():
    """动态计算最大线程数"""  # 新增函数
    cpu_count = psutil.cpu_count(logical=False)
    return min(100, (cpu_count * 4) + 4)


def test_proxy(proxy, timeout=PROXY_TIMEOUT):
    """优化后的代理测试函数"""  # 修改实现
    proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
    try:
        # 使用HEAD方法加速测试
        resp = requests.head(
            PROXY_TEST_URL,
            proxies=proxies,
            timeout=timeout,
            allow_redirects=False
        )
        return resp.status_code in [200, 302, 307]
    except:
        return False


def test_proxy_batch(proxies, timeout=PROXY_TIMEOUT):
    """批量测试代理"""  # 新增函数
    working = []
    with ThreadPoolExecutor(max_workers=auto_max_workers()) as executor:
        futures = {executor.submit(test_proxy, p, timeout): p for p in proxies}
        for future in tqdm(
                as_completed(futures),
                total=len(proxies),
                desc="Testing proxies",
                unit="proxy",
                ncols=100
        ):
            if future.result():
                working.append(futures[future])
    return working


def fetch_proxies():
    """重构后的代理获取函数"""  # 主要修改部分
    global proxy_pool, last_proxy_refresh

    print("[Proxy] Start updating proxy pool...")

    # 第一阶段：快速收集代理
    new_proxies = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(requests.get, url, timeout=10): url for url in PROXY_SOURCES}
        for future in as_completed(futures):
            try:
                if resp := future.result():
                    new_proxies.extend(resp.text.splitlines())
            except:
                continue

    # 预处理代理
    new_proxies = list(set([
        f"{p.split(':')[0]}:{p.split(':')[1]}"
        for p in new_proxies
        if len(p.split(':')) == 2
    ]))

    if not new_proxies:
        return

    # 第二阶段：快速初筛
    print("[Proxy] First stage testing...")
    stage1 = test_proxy_batch(new_proxies, timeout=2)

    # 第三阶段：精确验证
    print("[Proxy] Second stage testing...")
    stage2 = test_proxy_batch(stage1, timeout=5)

    with proxy_lock:
        proxy_pool = stage2
        last_proxy_refresh = time.time()

    print(f"[Proxy] Updated pool with {len(proxy_pool)} working proxies")


def get_random_proxy():
    """优化代理获取"""
    global last_proxy_refresh

    with proxy_lock:
        # 自动刷新逻辑
        if time.time() - last_proxy_refresh > PROXY_REFRESH_INTERVAL:
            threading.Thread(target=fetch_proxies).start()

        if proxy_pool:
            return random.choice(proxy_pool)
        return None




# 扩展材料科学术语（包含常见变体）
INCLUDE_KEYWORDS = [
    "In₂O₃", "TCO", "transparent conductive oxide",
    "ITO", "indium tin oxide", "Sn-doped In₂O₃",
    "doped indium oxide", "carrier concentration",
    "mobility enhancement", "sputtering deposition",
    "thin film conductivity", "optical transmittance",
    "figure of merit", "FOM", "Hall effect",
    "band gap engineering", "oxygen vacancy",
    "ALD", "chemical vapor deposition"
]

# 医学排除词（包含疾病和临床术语）
METAL_PROCESSING_KEYWORDS = [
    "metal forming", "forging", "stamping", "casting",
    "rolling", "extrusion", "sheet metal", "die casting",
    "machining", "cnc", "welding", "heat treatment"
]

# 生物医学扩展排除（新增子领域）
BIOMEDICAL_KEYWORDS = [
    "biocompatibility", "implant", "prosthesis", "orthopedic",
    "dental implant", "surgical instrument", "biopsy",
    "medical device", "pharmacokinetics", "clinical trial",
    "histology", "pathology", "radiology", "cell"
]

EXCLUDE_KEYWORDS = [
    # 排除光伏/太阳能电池领域
    "perovskite", "solar cell", "PV", "photovoltaic",
    "DSSC", "quantum dot sensitized",
    # 排除生物医学应用
    "biosensor", "biointerface", "implant",
    # 排除无关材料体系
    "ZnO", "TiO₂", "graphene", "carbon nanotube",
    # 原排除词保留
    *METAL_PROCESSING_KEYWORDS,
    *BIOMEDICAL_KEYWORDS
]
# 语义阈值
SEMANTIC_POS_THRESHOLD = 0.72
SEMANTIC_NEG_THRESHOLD = 0.28

# 期刊白名单
JOURNAL_WHITELIST = {
    'Journal of Materials Chemistry C',
    'Advanced Functional Materials',
    'Applied Physics Letters',
    'ACS Applied Materials & Interfaces',
    'Thin Solid Films',
    'Materials Science in Semiconductor Processing',
    'Journal of Applied Physics',
    'Ceramics International',
    'Optical Materials Express',
    'IEEE Transactions on Electron Devices',
    'Surface & Coatings Technology',
    'Journal of Vacuum Science & Technology A',
    'Materials Research Bulletin'
}
# 词干处理器
stemmer = PorterStemmer()
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ 模型初始化 ------------------
print(f"[{datetime.now()}] Loading semantic model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_membrane = model.encode("Synthetic polymer and ceramic membranes for molecular separation...",
                                  convert_to_tensor=True)

SEMANTIC_ANCHORS = {
    "tco_design": model.encode(
        "Design and optimization of Sn-doped In₂O₃ transparent conductive oxides "
        "with high conductivity and optical transparency. Focus on doping mechanisms, "
        "thin film deposition techniques (sputtering, ALD, CVD), and structure-property relationships.",
        convert_to_tensor=True
    ),
    "electrical_properties": model.encode(
        "Electrical characterization of TCO films including Hall effect measurements, "
        "carrier concentration optimization, and mobility enhancement strategies.",
        convert_to_tensor=True
    ),
    "optical_properties": model.encode(
        "Optical performance analysis of transparent conductive oxides in visible "
        "and near-infrared spectrum, band gap engineering, and FOM optimization.",
        convert_to_tensor=True
    )
}

embedding_medical = model.encode("Medical research including cancer diagnosis...", convert_to_tensor=True)

# ------------------ 数据库 ------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS literature (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT, doi TEXT UNIQUE, abstract TEXT,
    year INTEGER, journal TEXT, 
    is_whitelist BOOLEAN,
    pdf_url TEXT, save_path TEXT
)''')
conn.commit()


# ------------------ 带代理的请求函数 ------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def make_request(url, headers=None, timeout=REQUEST_TIMEOUT, allow_redirects=True):
    """带代理轮换和动态请求头的请求函数"""
    proxy = get_random_proxy()
    proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"} if proxy else None

    # 合并自定义headers和基础headers
    final_headers = {**get_random_headers(), **(headers or {})}

    try:
        resp = requests.get(
            url,
            headers=final_headers,
            proxies=proxies,
            timeout=timeout,
            allow_redirects=allow_redirects
        )
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        # 如果代理失败，尝试不使用代理
        if proxy:
            try:
                resp = requests.get(
                    url,
                    headers=final_headers,
                    timeout=timeout,
                    allow_redirects=allow_redirects
                )
                resp.raise_for_status()
                return resp
            except:
                raise e
        raise


# ------------------ 增强型重试策略 ------------------
CROSSREF_RETRY_SETTINGS = {
    'stop': stop_after_attempt(5),
    'wait': wait_exponential(multiplier=1, max=60),  # 指数退避策略
    'retry': retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.exceptions.ProxyError)),
    'before_sleep': lambda retry_state: print(
        f"[!] 第{retry_state.attempt_number}次重试，"
        f"错误：{type(retry_state.outcome.exception()).__name__}，"
        f"等待{retry_state.next_action.sleep:.1f}秒")
}

# ------------------ 核心改进：Crossref查询 ------------------
@retry(**CROSSREF_RETRY_SETTINGS)
def get_crossref_results(year, page):
    """服务器端关键词过滤"""
    base_query = " ".join([
        "In₂O₃", "TCO", "transparent conductive oxide",
        "ITO", "doped indium oxide", "Sn-doped"
    ])
    url = (
        f"https://api.crossref.org/works?"
        f"query.bibliographic={quote(base_query)}&"
        f"filter=from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        f",type:journal-article"
        f",has-full-text:true"
        f"&rows={ROWS_PER_PAGE}&offset={page * ROWS_PER_PAGE}"
    )

    headers = {
        "Accept": "application/json",
        "From": EMAIL
    }

    try:
        resp = make_request(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"[!] Crossref query failed: {str(e)[:80]}")
        raise


# ------------------ 关键词处理 ------------------
def preprocess_text(text):
    """标准化文本处理流程"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)  # 去除非字母数字字符
    words = [stemmer.stem(w) for w in text.split()]  # 词干提取
    return " ".join(words)


def contains_keywords(text, keywords):
    """带词干提取的关键词匹配"""
    processed_text = preprocess_text(text)
    return any(
        stemmer.stem(kw.lower()) in processed_text
        for kw in keywords
    )


# ------------------ 筛选逻辑 ------------------
def is_target_literature(item):
    """多级筛选函数"""
    journal = (item.get("container-title") or [""])[0]
    if journal in JOURNAL_WHITELIST:
        return True, True  # 返回(是否通过, 是否白名单)

    title = (item.get("title", [""])[0] or "")
    abstract = clean_html(item.get("abstract", ""))
    full_text = preprocess_text(f"{title} {abstract}")

    # 排除条件
    if contains_keywords(full_text, EXCLUDE_KEYWORDS):
        return False, False

    # 关键词计数
    include_count = sum(
        1 for kw in INCLUDE_KEYWORDS
        if stemmer.stem(kw.lower()) in full_text.split()
    )

    # 条件1：匹配至少3个关键词
    if include_count >= 3:
        return True, False

    # 条件2：语义验证
    embedding = model.encode(full_text, convert_to_tensor=True)
    return semantic_validation(embedding), False


# 语义验证函数
def semantic_validation(text_embedding):
    """动态语义阈值验证"""
    tco_sim = util.cos_sim(text_embedding, SEMANTIC_ANCHORS["tco_design"]).item()
    elec_sim = util.cos_sim(text_embedding, SEMANTIC_ANCHORS["electrical_properties"]).item()
    opt_sim = util.cos_sim(text_embedding, SEMANTIC_ANCHORS["optical_properties"]).item()

    return (
            (tco_sim >= 0.75) or
            (elec_sim + opt_sim > 1.4) and
            (tco_sim > 0.65)
    )
# ------------------ 增强下载验证 ------------------


def is_valid_pdf(filepath):
    """验证PDF文件真实性"""
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'%PDF'
    except:
        return False


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf(pdf_url, doi):
    """来源验证"""
    try:
        # 根据来源设置不同headers
        if "sciencedirect" in pdf_url:
            headers = {"Accept": "application/pdf", "Referer": "https://www.sciencedirect.com/"}
        elif "springer" in pdf_url:
            headers = {"Accept": "application/pdf", "Referer": "https://link.springer.com/"}
        else:
            headers = {"Accept": "application/pdf"}

        resp = make_request(pdf_url, headers=headers, timeout=20)

        if resp.ok and b"%PDF" in resp.content[:1024]:
            filename = f"{doi.replace('/', '_')}.pdf"
            save_path = os.path.join(SAVE_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return save_path if is_valid_pdf(save_path) else None
    except Exception as e:
        print(f"[!] Download failed ({doi}): {str(e)[:50]}")
    return None


def clean_html(raw_text):
    """HTML清洗"""
    clean_text = re.sub(r"<[^>]+>", "", raw_text or "")
    clean_text = re.sub(r"\s+", " ", unescape(clean_text)).strip()
    return clean_text[:2000]  # 限制摘要长度


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_pdf_url_from_unpaywall(doi):
    """空数据处理"""
    try:
        url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={EMAIL}"
        headers = {"Accept": "application/json"}

        resp = make_request(url, headers=headers, timeout=15)

        # HTTP状态码处理
        if resp.status_code == 404:
            print(f"[i] DOI {doi} not found in Unpaywall")
            return None
        if resp.status_code != 200:
            print(f"[!] Unpaywall API error: HTTP {resp.status_code}")
            return None

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"[!] Invalid JSON response for DOI {doi}")
            return None

        # 空数据保护
        if not isinstance(data, dict):
            return None

        # 优先选择出版商官方链接
        best_oa = data.get("best_oa_location", {}) or {}
        if best_oa.get("host_type") == "publisher":
            return best_oa.get("url_for_pdf")

        # 次选其他合法来源
        for loc in data.get("oa_locations", []):
            if isinstance(loc, dict) and (url := loc.get("url_for_pdf")):
                return url

        return None

    except Exception as e:
        print(f"[!] Unpaywall critical error: {str(e)[:80]}")
        return None


# ------------------ 新增：Sci-Hub备用方案 ------------------
def find_scihub_url(doi):
    """多域名尝试策略"""
    domains = [
        "https://sci-hub.se",
        "https://www.materialscloud.org",
        "https://iopscience.iop.org"
    ]

    for domain in domains:
        try:
            url = f"{domain}/{doi}"
            resp = make_request(url, timeout=10, allow_redirects=True)
            if resp.status_code < 400:
                return url
        except:
            continue
    return None


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        fetch_proxies()

        for year in range(START_YEAR, END_YEAR + 1):
            print(f"\n[{datetime.now()}] Processing year {year}")

            page_retry_count = 0  # 页面级重试计数器
            current_page = 0  # 当前处理页码
            max_retries = 3  # 单页最大重试次数


            with tqdm(total=MAX_PAGES, desc=f"Pages {year}", unit="page") as pbar:
                while current_page < MAX_PAGES:
                    try:
                        items = get_crossref_results(year, current_page)

                        # 检查是否没有更多数据
                        if not items:
                            print(f"[√] Year {year} data complete at page {current_page}")
                            break

                        # 处理当前页数据
                        valid_count = 0
                        for item in items:
                            doi = item.get("DOI")
                            if not doi:
                                continue

                            # 去重检查
                            cursor.execute("SELECT 1 FROM literature WHERE doi=?", (doi,))
                            if cursor.fetchone():
                                continue

                            # 筛选逻辑
                            is_valid, is_whitelist = is_target_literature(item)
                            if not is_valid:
                                continue

                            # 获取PDF链接
                            pdf_url = get_pdf_url_from_unpaywall(doi) or find_scihub_url(doi)
                            if not pdf_url:
                                continue

                            # 下载并保存
                            if (save_path := download_pdf(pdf_url, doi)):
                                cursor.execute('''
                                    INSERT INTO literature 
                                    (title, doi, abstract, year, journal, is_whitelist, pdf_url, save_path)
                                    VALUES (?,?,?,?,?,?,?,?)
                                ''', (
                                    (item.get("title") or [""])[0],
                                    doi,
                                    clean_html(item.get("abstract")),
                                    (item.get("issued", {}).get("date-parts", [[None]])[0][0]),
                                    (item.get("container-title") or [""])[0],
                                    is_whitelist,
                                    pdf_url,
                                    save_path
                                ))
                                valid_count += 1

                        # 成功处理当前页
                        conn.commit()
                        pbar.update(1)  # 更新进度条
                        current_page += 1
                        page_retry_count = 0  # 重置重试计数器
                        time.sleep(API_CALL_INTERVAL)  # 正常请求间隔

                    except Exception as e:
                        error_msg = f"[!] Year {year} Page {current_page} error ({page_retry_count + 1}/{max_retries}): {str(e)[:100]}"

                        # 分级处理策略
                        if page_retry_count < max_retries:
                            print(error_msg + " → Retrying...")
                            sleep_time = 10 * (2 ** page_retry_count)  # 指数退避
                            time.sleep(sleep_time)
                            page_retry_count += 1
                        else:
                            print(error_msg + " → Skip page")
                            current_page += 1  # 强制推进页码
                            pbar.update(1)
                            page_retry_count = 0
                            time.sleep(30)

                        # 代理相关错误处理
                        if isinstance(e, requests.exceptions.ProxyError):
                            print("[!] Detected proxy issues, refreshing proxy pool...")
                            fetch_proxies()

                        # 增加对RetryError的特殊处理
                        if isinstance(e, tenacity.RetryError):
                            print("[!] Critical retry failure, pausing 3 minutes...")
                            time.sleep(180)

                        # 其他错误处理
                        if isinstance(e, requests.Timeout):
                            print("[!] Timeout detected, adjusting request parameters...")
                            time.sleep(60)



    finally:
        conn.close()
        print(f"\n[{datetime.now()}] Process completed!")