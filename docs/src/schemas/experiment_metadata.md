(readmeexp)=
# Experiment metadata

**Global metadata table**

|             key           | description |
|---------------------------|-----------------------------------------|
| ARTICLE_DOI | DOI of the of the original publication of the experimental data|
| DATA_DOI | DOI of the dataset deposition with raw NMR data |
| DATA_REF | Reference to deposited dataset |
| TEMPERATURE | Temperature (K) of the experiment |
| MEMBRANE_COMPOSITION | Dictionary of molar fractions of membrane phase |
| SOLUTION_COMPOSITION | Dictionary of ion concentrations in the system |
| ADDITIONAL_MOLECULES| Dictionary of molecules which are not in the databank |
| PH | pH of the system |
| PH_METHOD | Method of pH setting or measurement (buffer / measurement) |
| REAGENT_SOURCES | Description of lipid reagents. Source, purity, etc. |

**NMR specific metadata**

|             key           | description                             |
|---------------------------|-----------------------------------------|
| T_RF_HEATING | Correction of the temperature according to RF heating |
| TOTAL_HYDRATION | Total hydration of the system (water mass %) |
| HYDRATION_METHOD | Method of preparing lamellar phase |
| NMR_INSTRUMENT | Instrument name (including field) |
| NMR_METHOD | Method of OP measurement "abbr1:abbr2" (see detailed explaining below) |
| SIGN_MEASURED | NONE or S-DROSS |
| NMR_COMMENT | Description of NMR experiment type |

**X-ray specific metadata**

|             key           | description                             |
|---------------------------|-----------------------------------------|
| XR_SOURCE | Source name |
| XR_LAMBDA | Source wavelength or range |
| XR_QRANGE | Scattering detection range (Q-range) |
| XR_DETECTOR | Detector type |
| XR_DISTANCE | Distance to detector (mm) |
| XR_EXPOSITION | Exposition time |
| SAMPLE_TYPE | 'MLV', 'SUV', 'GUV', 'OS' (oriented sample) |
| SAMPLE_PROTOCOL | Protocol of liposome (or OS) preparation |

## General fields

1. **ARTICLE_DOI**  
DOI of the original publication where the experimental data originates.

2. **DATA_DOI**  
DOI of the dataset deposition with raw NMR data (e.g., nmrXive).

3. **DATA_REF**
If the dataset doesn't have DOI, we engage to add some persistent identifier or even URL if the first doesn't exist.

3. **TEMPERATURE**  
Temperature (K) of the experiment. If `T_RF_HEATING` is 'no_inromation', the reported temperature from the probe is given. Otherwise, please insert RF-corrected temperature.

4. **T_RF_HEATING**  
How RF heating is dealt (UNKNOWN / measured / guessed)

5. **MEMBRANE_COMPOSITION**  
Dictionary of molar fractions of bilayer components. For example:
```
MEMBRANE_COMPOSITION:
  POPC: 0.93
  CHOL: 0.07
```
All the molecules should be registered in the [molecular inventory](molecule_record) in the ``membrane`` subfolder.

6. **SOLUTION_COMPOSITION**  
Dictionary of solution composition of the system (mass %), main solvent is not listed:
```
SOLUTION_COMPOSITION:
  SOD: 0.5
  CLA: 0.24
  GLUCOSE: 0.1
```
All the molecules should be registered in the [molecular inventory](molecule_record) in the ``solution`` subfolder.
Do not provide whole salts! Only separated ions. Remember that the counterions of charged lipids are also part of the solution.

7. **ADDITIONAL_MOLECULES**
Dictionary of additional molecules in the format:
```
ADDITIONAL_MOLECULES:
    TFA: trifluoroacetic acid, 0.1%
    DMSO: dimethylsulfoxide, 0.1%
    DSS: sodium trimethylsilylpropanesulfonate, 0.01%
    EDTA: ethylenediaminetetraacetic acid, 0.1 mM
```
we can use INCHI-key, CAS number or just IUPAC name. If molecule is important for
the composition, it should get the metadata inside the databank and be mentioned under
`SOLUTION_COMPOSITION` instead.

8. **PH**  
pH of the system (number or UNKNOWN)

9. **PH_METHOD**  
How the pH value is got: measured by pH electrode or indicator paper, measured by NMR, set by buffer.

10. **REAGENT_SOURCES**  
Which reagents are used for lipids -- should be specified for every lipid.

## NMR-specific fields

1. **TOTAL_HYDRATION**  
Mass \% of water. Better if it is measured by <sup>1</sup>H MAS NMR.

2. **HYDRATION_METHOD**  
Way how the targeted hydration level is reached: lyophilised powder is hydrated, liposome suspension is dehydrated, or liposome suspension is ultracentrifugated to get lipid-rich phase.

3. **NMR_INSTRUMENT**  
Name of the instrument and field strength.

4. **NMR_METHOD**  
A field identifying the NMR method used (string formed as METHOD:SUBMETHOD, e.g., "2H:QE").
    - Variants for METHOD: *"2H", "CDLF", "PDLF"*  
      Two main methods are <sup>2</sup>H-NMR and <sup>1</sup>H-<sup>13</sup>C SLF (separate local field)
      NMR experiments which can be either CDLF (Carbon-detected local field) or PDLF (Proton-DLF).  
    - Sub-Method for "2H": *"SP" | "QE" | "see_comments"*  
      For <sup>2</sup>H NMR, the submethod used is either "single pulse" or "quadrupolar echo".
    - Sub-Method for "CDLF": *"REDOR" | "DIPSHIFT" | "recDIPSHIFT" | "see_comments"*  
      For CDLF method, the variants could be Rotational-Echo Double-Resonance (REDOR),
      Dipolar-Coupling chemical shift correlation (DIPSHIFT), or recoupled DIPSHIFT (recDIPSHIFT).
    - Sub-Method for "PDLF": *"DROSS" | "R18_1^7" (or other numbers characterizing R-type sequence) | "see_comments"*  
      For PDLF method, subvariants could use dipolar recoupling on-axis with scaling and shape preservation (DROSS),
      or R-type recoupling (recoupling using symmetry-based pulse sequences)

5. **SIGN_MEASURED**  
Method name  (e.g. S-DROSS) if order parameter sign was measured, NONE otherwise.

6. **NMR_COMMENTS**  
Links to the pulse sequence, corresponding paper and precise parameters if important.
Obligatory explanation if **NMR_METHOD** uses "see_comments" for SUBMETHOD.

## Scattering-specific fields

1. **XR_SOURCE**
X-ray source description. Name of the core facilities or instrument name if laboratory source was used.

2. **XR_LAMBDA**
Source wavelength or range. Wavelength (and/or range) of the X-ray beam used, with units (e.g., Angstroms).

3. **XR_QRANGE**
Scattering detection range (Q-range). The accessible scattering vector range, typically given in 1/Å.

4. **XR_DETECTOR**
Detector type. The detector installed (e.g., CCD camera, PILATUS, or other detector model).

5. **XR_DISTANCE**
Distance to detector (mm). The separation between the sample and the detector, in millimeters.

6. **XR_EXPOSITION**
Exposition time. The total data acquisition time per measurement, usually in seconds.

7. **SAMPLE_TYPE**
'MLV', 'SUV', 'GUV', 'OS' (oriented sample). The type of sample used, with definitions: MLV (multilamellar vesicles), SUV (small unilamellar vesicles), GUV (giant unilamellar vesicles), OS (oriented sample).

8. **SAMPLE_PROTOCOL**
Protocol of liposome (or OS) preparation. A description of the preparation steps and conditions used to obtain the sample, such as lipid composition, hydration method, extrusion, alignment procedures, and any buffers used.