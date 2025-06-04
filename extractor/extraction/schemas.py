"""
Module: schemas
Functionality: Defines Pydantic models for validating the structure and data types
               of the information extracted by the LLM. This helps ensure data
               consistency and quality before it's saved or used in downstream tasks.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, RootModel

# --- Base Schema Configuration ---
class BaseSchema(BaseModel):
    """Base schema with common configurations for TCO models."""
    model_config = {
        "extra": "allow",  # Allow extra fields not explicitly defined
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }

# --- Detailed Sub-Schemas for TCOs ---

class TCODopantSchema(BaseSchema):
    element: Optional[str] = Field(default=None, description="Symbol or name of the dopant element (e.g., 'Sn', 'W', 'Mo').")
    concentration_value: Optional[float] = Field(default=None, description="Numerical value of doping concentration.")
    concentration_unit: Optional[str] = Field(default=None, description="Unit of doping concentration (e.g., 'at.%', 'wt.%', 'mol%').")
    concentration_text: Optional[str] = Field(default=None, description="Full doping concentration as text, e.g., '5 at.%', '1-3 wt.%'. Use if value/unit parsing is complex.")

class TCODesignSchema(BaseSchema):
    host_material: Optional[str] = Field(default="In2O3", alias="HostMaterial", description="Primary host oxide material, typically In2O3.")
    primary_dopant: Optional[TCODopantSchema] = Field(default_factory=TCODopantSchema, alias="PrimaryDopant", description="Main dopant details.")
    co_dopants: Optional[List[TCODopantSchema]] = Field(default_factory=list, alias="CoDopants", description="List of co-dopants, if any.")
    target_stoichiometry: Optional[str] = Field(default=None, alias="TargetStoichiometry", description="Intended stoichiometric formula, e.g., 'In(2-x)SnxO3'.")
    composite_materials: Optional[str] = Field(default=None, alias="CompositeMaterials", description="Other materials in a composite structure, if applicable.")
    material_description_source: Optional[str] = Field(default=None, alias="MaterialDescriptionSource", description="Source of the material if commercial, e.g., 'Commercial ITO target (99.99% purity)'.")

class TCODepositionParamsSchema(BaseSchema):
    base_pressure: Optional[str] = Field(default=None, alias="BasePressure", description="Base pressure before deposition, e.g., '1x10^-6 Torr'.")
    working_pressure: Optional[str] = Field(default=None, alias="WorkingPressure", description="Working pressure during deposition, e.g., '3 mTorr'.")
    deposition_temperature: Optional[str] = Field(default=None, alias="DepositionTemperature", description="Substrate temperature during deposition, e.g., '300°C', 'Room Temperature'.")
    target_substrate_distance: Optional[str] = Field(default=None, alias="TargetSubstrateDistance", description="Distance between target and substrate.")
    gas_atmosphere: Optional[str] = Field(default=None, alias="GasAtmosphere", description="Gases used during deposition, e.g., 'Ar', 'Ar/O2 mixture'.")
    gas_flow_rates: Optional[str] = Field(default=None, alias="GasFlowRates", description="Flow rates of gases, e.g., 'Ar: 20 sccm, O2: 1 sccm'.")
    deposition_power: Optional[str] = Field(default=None, alias="DepositionPower", description="Power applied for sputtering/PLD, e.g., '100 W (DC)', '150 mJ/pulse'.")
    deposition_time: Optional[str] = Field(default=None, alias="DepositionTime", description="Duration of the deposition process.")
    deposition_rate: Optional[str] = Field(default=None, alias="DepositionRate", description="Rate of film growth, e.g., '0.5 nm/s'.")
    # Add other method-specific parameters (e.g., precursor details for ALD/CVD, laser parameters for PLD)

class TCOAnnealingSchema(BaseSchema):
    temperature: Optional[str] = Field(default=None, alias="Temperature", description="Annealing temperature, e.g., '500°C'.")
    atmosphere: Optional[str] = Field(default=None, alias="Atmosphere", description="Annealing atmosphere, e.g., 'N2', 'Vacuum', 'Forming Gas (5% H2/95% N2)'.")
    duration: Optional[str] = Field(default=None, alias="Duration", description="Annealing duration, e.g., '1 hour', '30 min'.")
    ramp_rate: Optional[str] = Field(default=None, alias="RampRate", description="Heating/cooling ramp rate if specified.")

class TCOFabricationSchema(BaseSchema):
    deposition_method: Optional[str] = Field(default=None, alias="DepositionMethod", description="Primary method used for film deposition.")
    substrate_material: Optional[str] = Field(default=None, alias="SubstrateMaterial", description="Material of the substrate, e.g., 'Glass', 'PET', 'Si'.")
    precursor_materials_text: Optional[str] = Field(default=None, alias="PrecursorMaterialsText", description="Precursor materials used (for sol-gel, CVD, ALD), as text.")
    target_material_text: Optional[str] = Field(default=None, alias="TargetMaterialText", description="Target material composition (for sputtering, PLD), as text.")
    deposition_parameters: Optional[TCODepositionParamsSchema] = Field(default_factory=TCODepositionParamsSchema, alias="DepositionParameters")
    deposition_parameters_text_summary: Optional[str] = Field(default=None, alias="DepositionParametersTextSummary", description="General descriptive text if specific deposition parameters are not itemized.") # <<< NEW FIELD
    annealing_conditions: Optional[TCOAnnealingSchema] = Field(default_factory=TCOAnnealingSchema, alias="AnnealingConditions")
    annealing_conditions_text_summary: Optional[str] = Field(default=None, alias="AnnealingConditionsTextSummary", description="General descriptive text if specific annealing conditions are not itemized.") # <<< NEW FIELD
    film_thickness_value: Optional[float] = Field(default=None, alias="FilmThicknessValue", description="Numerical value of film thickness.")
    film_thickness_unit: Optional[str] = Field(default="nm", alias="FilmThicknessUnit", description="Unit of film thickness (e.g., 'nm', 'µm').")
    film_thickness_text: Optional[str] = Field(default=None, alias="FilmThicknessText", description="Full film thickness as text, e.g., '150 nm', 'approx. 200 nm'.")
    # Any other general fabrication notes

class TCOElectricalPropertiesSchema(BaseSchema):
    resistivity: Optional[str] = Field(default=None, alias="Resistivity", description="Electrical resistivity, e.g., '3.5 x 10^-4 Ω·cm'.")
    sheet_resistance: Optional[str] = Field(default=None, alias="SheetResistance", description="Sheet resistance, e.g., '10 Ω/sq'.")
    conductivity: Optional[str] = Field(default=None, alias="Conductivity", description="Electrical conductivity, e.g., '2857 S/cm'.")
    carrier_concentration: Optional[str] = Field(default=None, alias="CarrierConcentration", description="Carrier concentration, e.g., '6.2 x 10^20 cm⁻³'.")
    carrier_type: Optional[str] = Field(default=None, alias="CarrierType", description="Predominant carrier type, e.g., 'n-type', 'p-type'.")
    hall_mobility: Optional[str] = Field(default=None, alias="HallMobility", description="Hall mobility, e.g., '45 cm²/Vs'.")
    measurement_conditions_electrical: Optional[str] = Field(default=None, alias="MeasurementConditionsElectrical", description="Conditions under which electrical properties were measured, e.g., 'Room temperature, Van der Pauw method'.")

class TCOOpticalPropertiesSchema(BaseSchema):
    average_transmittance: Optional[str] = Field(default=None, alias="AverageTransmittance", description="Average optical transmittance, e.g., '>90% in visible range (400-700nm)'.")
    transmittance_at_550nm: Optional[str] = Field(default=None, alias="TransmittanceAt550nm", description="Transmittance at specific wavelength, e.g., '92% at 550 nm'.")
    wavelength_range_transmittance: Optional[str] = Field(default=None, alias="WavelengthRangeTransmittance", description="Wavelength range for transmittance measurement.")
    optical_band_gap_value: Optional[float] = Field(default=None, alias="OpticalBandGapValue", description="Numerical value of optical band gap (Eg).")
    optical_band_gap_unit: Optional[str] = Field(default="eV", alias="OpticalBandGapUnit", description="Unit for optical band gap, typically 'eV'.")
    optical_band_gap_text: Optional[str] = Field(default=None, alias="OpticalBandGapText", description="Full optical band gap as text, e.g., '3.75 eV'.")
    band_gap_determination_method: Optional[str] = Field(default=None, alias="BandGapDeterminationMethod", description="Method used for band gap calculation, e.g., 'Tauc plot'.")
    refractive_index: Optional[str] = Field(default=None, alias="RefractiveIndex", description="Refractive index, possibly at a specific wavelength.")
    extinction_coefficient: Optional[str] = Field(default=None, alias="ExtinctionCoefficient", description="Extinction coefficient (k).")
    haze: Optional[str] = Field(default=None, alias="Haze", description="Haze percentage, e.g., '< 1%'.")

class TCOOtherPerformanceSchema(BaseSchema):
    work_function_value: Optional[float] = Field(default=None, alias="WorkFunctionValue", description="Numerical value of work function (Φ).")
    work_function_unit: Optional[str] = Field(default="eV", alias="WorkFunctionUnit", description="Unit for work function, typically 'eV'.")
    work_function_text: Optional[str] = Field(default=None, alias="WorkFunctionText", description="Full work function as text, e.g., '4.8 eV'.")
    work_function_method: Optional[str] = Field(default=None, alias="WorkFunctionMethod", description="Method for work function measurement, e.g., 'UPS', 'Kelvin Probe'.")
    figure_of_merit_value: Optional[str] = Field(default=None, alias="FigureOfMeritValue", description="Value of the Figure of Merit.") # Can be complex string
    figure_of_merit_type: Optional[str] = Field(default=None, alias="FigureOfMeritType", description="Type of Figure of Merit, e.g., 'Haacke', 'Fraunhofer'.")
    surface_roughness_rms: Optional[str] = Field(default=None, alias="SurfaceRoughnessRMS", description="Root Mean Square surface roughness, e.g., '0.5 nm (RMS)'.")
    surface_roughness_ra: Optional[str] = Field(default=None, alias="SurfaceRoughnessRa", description="Average surface roughness (Ra).")
    stability: Optional[str] = Field(default=None, alias="Stability", description="Notes on material stability (thermal, chemical, environmental).")

class TCOStructuralPropertiesSchema(BaseSchema):
    crystal_structure: Optional[str] = Field(default=None, alias="CrystalStructure", description="Crystal structure and phases identified, e.g., 'Cubic bixbyite In2O3', 'Amorphous'.")
    preferred_orientation: Optional[str] = Field(default=None, alias="PreferredOrientation", description="Preferred crystallographic orientation, e.g., '(222)', '(400)'.")
    grain_size: Optional[str] = Field(default=None, alias="GrainSize", description="Average grain size, e.g., '30-50 nm'.")
    surface_morphology_notes: Optional[str] = Field(default=None, alias="SurfaceMorphologyNotes", description="Observations from SEM, AFM, etc.")
    xrd_notes: Optional[str] = Field(default=None, alias="XRDNotes", description="Key findings from XRD analysis.")


class TCOPerformanceSchema(BaseSchema):
    electrical_properties: Optional[TCOElectricalPropertiesSchema] = Field(default_factory=TCOElectricalPropertiesSchema, alias="ElectricalProperties")
    optical_properties: Optional[TCOOpticalPropertiesSchema] = Field(default_factory=TCOOpticalPropertiesSchema, alias="OpticalProperties")
    structural_properties: Optional[TCOStructuralPropertiesSchema] = Field(default_factory=TCOStructuralPropertiesSchema, alias="StructuralProperties")
    other_performance_metrics: Optional[TCOOtherPerformanceSchema] = Field(default_factory=TCOOtherPerformanceSchema, alias="OtherPerformanceMetrics")
    general_performance_summary: Optional[str] = Field(default=None, alias="GeneralPerformanceSummary", description="A general summary if specific values are not broken down.")


class TCODeviceSchema(BaseSchema):
    device_type: Optional[str] = Field(default=None, alias="DeviceType", description="Type of device where TCO is used, e.g., 'Perovskite Solar Cell', 'OLED'.")
    tco_role_in_device: Optional[str] = Field(default=None, alias="TCORoleInDevice", description="Role of the TCO in the device, e.g., 'Anode', 'Transparent Top Electrode'.")
    device_performance_metric_name: Optional[str] = Field(default=None, alias="DevicePerformanceMetricName", description="Name of the key device performance metric, e.g., 'Power Conversion Efficiency (PCE)', 'External Quantum Efficiency (EQE)'.")
    device_performance_metric_value: Optional[str] = Field(default=None, alias="DevicePerformanceMetricValue", description="Value of the key device performance metric, e.g., '18.5%', '25 cd/A'.")
    other_device_details: Optional[str] = Field(default=None, alias="OtherDeviceDetails", description="Other relevant details about the device construction or performance.")

class TCOApplicationSchema(BaseSchema):
    potential_application_area: Optional[str] = Field(default=None, alias="PotentialApplicationArea", description="General field of application, e.g., 'Photovoltaics', 'Flexible Electronics'.")
    demonstrated_in_device: Optional[TCODeviceSchema] = Field(default_factory=TCODeviceSchema, alias="DemonstratedInDevice")
    # If multiple devices, this could be List[TCODeviceSchema] but prompt asks for one entry per "sample"
    # so this might be specific to the device that *this particular sample* was tested in.

class TCOSpecificDetailsSchema(BaseSchema): # This is what goes into "Details" for a TCO entry
    Design: Optional[TCODesignSchema] = Field(default_factory=TCODesignSchema)
    Fabrication: Optional[TCOFabricationSchema] = Field(default_factory=TCOFabricationSchema)
    Performance: Optional[TCOPerformanceSchema] = Field(default_factory=TCOPerformanceSchema)
    Application: Optional[TCOApplicationSchema] = Field(default_factory=TCOApplicationSchema)

    @model_validator(mode='before')
    @classmethod
    def ensure_detail_sections_are_dicts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for section_name in ["Design", "Fabrication", "Performance", "Application"]:
                # If LLM provides a section as `null` or not a dict, convert it to an empty dict
                # so Pydantic can initialize the respective schema with defaults.
                if data.get(section_name) is None or not isinstance(data.get(section_name), dict):
                    data[section_name] = {}
        return data


# --- Schemas for others ---
# ........................
# ........................
# ........................
# ........................



# --- Main Schemas for LLM Output and Individual Entries ---
class ExtractedMaterialEntrySchema(BaseSchema):
    """Schema for a single extracted material entry (can be TCO or Membrane)."""
    # This should match the "MaterialName" key from your new TCO prompt
    MaterialName: str = Field(..., min_length=1, description="Descriptive name of the extracted material/sample.")
    # The 'Details' field will be validated more specifically by domain in the parser
    Details: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific details of the material.")


class LLMOutputSchema(BaseModel): # No BaseTCOSchema here as it's the root
    """
    Defines the expected root structure of the JSON returned by the LLM,
    after initial key standardization (e.g., "output" to "Output").
    """
    model_config = { "str_strip_whitespace": True } # Root model config

    Output: List[ExtractedMaterialEntrySchema] = Field(default_factory=list, description="List of extracted material entries from the paper.")

    @model_validator(mode='before')
    @classmethod
    def ensure_output_is_list_and_not_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            output_val = data.get('Output') # Check standardized key "Output"
            if output_val is None: # If key "Output" exists but value is None
                data['Output'] = []
            elif not isinstance(output_val, list):
                # If LLM returns a single object instead of a list for Output
                data['Output'] = [output_val]
            # If 'Output' key is entirely missing, Pydantic will raise error unless default_factory handles it, which it does.
        elif data is None: # If the entire data object passed to LLMOutputSchema is None
            return {"Output": []} # Return a valid structure
        return data