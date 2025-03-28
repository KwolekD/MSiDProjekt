import kagglehub,sys,os,shutil
import pandas as pd
import numpy as np

def changeFromNumericalToCategorical(dataf: pd.DataFrame)->pd.DataFrame:
    resultDataFrame = dataf.copy()
    marital_status = {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated"
    }
    
    resultDataFrame["Marital status"] = resultDataFrame["Marital status"].map(marital_status)
    resultDataFrame["Marital status"] = resultDataFrame["Marital status"].astype("category")

    application_modes = {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        3: "1st phase - special contingent (Azores Island)",
        4: "Holders of other higher courses",
        5: "Ordinance No. 854-B/99",
        6: "International student (bachelor)",
        7: "1st phase - special contingent (Madeira Island)",
        8: "2nd phase - general contingent",
        9: "3rd phase - general contingent",
        10: "Ordinance No. 533-A/99, item b2) (Different Plan)",
        11: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        12: "Over 23 years old",
        13: "Transfer",
        14: "Change of course",
        15: "Technological specialization diploma holders",
        16: "Change of institution/course",
        17: "Short cycle diploma holders",
        18: "Change of institution/course (International)"
    }

    resultDataFrame["Application mode"] = resultDataFrame["Application mode"].map(application_modes)
    resultDataFrame["Application mode"] = resultDataFrame["Application mode"].astype("category")
    
    courses = {
        1: "Biofuel Production Technologies",
        2: "Animation and Multimedia Design",
        3: "Social Service (evening attendance)",
        4: "Agronomy",
        5: "Communication Design",
        6: "Veterinary Nursing",
        7: "Informatics Engineering",
        8: "Equinculture", 
        9: "Management",
        10: "Social Service",
        11: "Tourism",
        12: "Nursing",
        13: "Oral Hygiene",
        14: "Advertising and Marketing Management",
        15: "Journalism and Communication",
        16: "Basic Education",
        17: "Management (evening attendance)"
    }

    resultDataFrame["Course"] = resultDataFrame["Course"].map(courses)
    resultDataFrame["Course"] = resultDataFrame["Course"].astype("category")

    day_part = {
        1: "Daytime",
        0: "Evening"
    }

    resultDataFrame["Daytime/evening attendance"] = resultDataFrame["Daytime/evening attendance"].map(day_part)
    resultDataFrame["Daytime/evening attendance"] = resultDataFrame["Daytime/evening attendance"].astype("category")
    
    prev_education_dict = education_dict = {
        1: "Secondary education",
        2: "Higher education - Bachelor's Degree",
        3: "Higher education - Degree",
        4: "Higher education - Master's",
        5: "Higher education - Doctorate",
        6: "Frequency of higher education",
        7: "12th year of schooling - Not completed",
        8: "11th year of schooling - Not completed",
        9: "Other - 11th year of schooling",
        10: "10th year of schooling",
        11: "10th year of schooling - Not completed",
        12: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
        13: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
        14: "Technological specialization course",
        15: "Higher education - Degree (1st cycle)",
        16: "Professional higher technical course",
        17: "Higher education - Master (2nd cycle)"
    }

    resultDataFrame["Previous qualification"] = resultDataFrame["Previous qualification"].map(prev_education_dict)
    resultDataFrame["Previous qualification"] = resultDataFrame["Previous qualification"].astype("category")

    nationality_dict = {
        1: "Portuguese",
        2: "German",
        3: "Spanish",
        4: "Italian",
        5: "Dutch",
        6: "English",
        7: "Lithuanian",
        8: "Angolan",
        9: "Cape Verdean",
        10: "Guinean",
        11: "Mozambican",
        12: "Santomean",
        13: "Turkish",
        14: "Brazilian",
        15: "Romanian",
        16: "Moldova (Republic of)",
        17: "Mexican",
        18: "Ukrainian",
        19: "Russian",
        20: "Cuban",
        21: "Colombian"
    }

    resultDataFrame["Nacionality"] = resultDataFrame["Nacionality"].map(nationality_dict)
    resultDataFrame["Nacionality"] = resultDataFrame["Nacionality"].astype("category")

    education_dict_mother = {
        1: "Secondary Education - 12th Year of Schooling or Eq.",
        2: "Higher Education - Bachelor's Degree",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        7: "12th Year of Schooling - Not Completed",
        8: "11th Year of Schooling - Not Completed",
        9: "7th Year (Old)",
        10: "Other - 11th Year of Schooling",
        11: "10th Year of Schooling",
        12: "General commerce course",
        13: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
        14: "Technical-professional course",
        15: "7th year of schooling",
        16: "2nd cycle of the general high school course",
        17: "9th Year of Schooling - Not Completed",
        18: "8th year of schooling",
        19: "Unknown",
        20: "Can't read or write",
        21: "Can read without having a 4th year of schooling",
        22: "Basic education 1st cycle (4th/5th year) or equiv.",
        23: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
        24: "Technological specialization course",
        25: "Higher education - degree (1st cycle)",
        26: "Specialized higher studies course",
        27: "Professional higher technical course",
        28: "Higher Education - Master (2nd cycle)",
        29: "Higher Education - Doctorate (3rd cycle)"
    }

    resultDataFrame["Mother's qualification"] = resultDataFrame["Mother's qualification"].map(education_dict_mother)
    resultDataFrame["Mother's qualification"] = resultDataFrame["Mother's qualification"].astype("category")

    education_dict_father = {
        1: "Secondary Education - 12th Year of Schooling or Eq.",
        2: "Higher Education - Bachelor's Degree",
        3: "Higher Education - Degree",
        4: "Higher Education - Master's",
        5: "Higher Education - Doctorate",
        6: "Frequency of Higher Education",
        7: "12th Year of Schooling - Not Completed",
        8: "11th Year of Schooling - Not Completed",
        9: "7th Year (Old)",
        10: "Other - 11th Year of Schooling",
        11: "2nd year complementary high school course",
        12: "10th Year of Schooling",
        13: "General commerce course",
        14: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
        15: "Complementary High School Course",
        16: "Technical-professional course",
        17: "Complementary High School Course - not concluded",
        18: "7th year of schooling",
        19: "2nd cycle of the general high school course",
        20: "9th Year of Schooling - Not Completed",
        21: "8th year of schooling",
        22: "General Course of Administration and Commerce",
        23: "Supplementary Accounting and Administration",
        24: "Unknown",
        25: "Can't read or write",
        26: "Can read without having a 4th year of schooling",
        27: "Basic education 1st cycle (4th/5th year) or equiv.",
        28: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
        29: "Technological specialization course",
        30: "Higher education - degree (1st cycle)",
        31: "Specialized higher studies course",
        32: "Professional higher technical course",
        33: "Higher Education - Master (2nd cycle)",
        34: "Higher Education - Doctorate (3rd cycle)"
    }

    resultDataFrame["Father's qualification"] = resultDataFrame["Father's qualification"].map(education_dict_father)
    resultDataFrame["Father's qualification"] = resultDataFrame["Father's qualification"].astype("category")

    occupation_dict_mother = {
        1: "Student",
        2: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
        3: "Specialists in Intellectual and Scientific Activities",
        4: "Intermediate Level Technicians and Professions",
        5: "Administrative staff",
        6: "Personal Services, Security and Safety Workers and Sellers",
        7: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
        8: "Skilled Workers in Industry, Construction and Craftsmen",
        9: "Installation and Machine Operators and Assembly Workers",
        10: "Unskilled Workers",
        11: "Armed Forces Professions",
        12: "Other Situation",
        13: "(blank)",
        14: "Health professionals",
        15: "Teachers",
        16: "Specialists in information and communication technologies (ICT)",
        17: "Intermediate level science and engineering technicians and professions",
        18: "Technicians and professionals, of intermediate level of health",
        19: "Intermediate level technicians from legal, social, sports, cultural and similar services",
        20: "Office workers, secretaries in general and data processing operators",
        21: "Data, accounting, statistical, financial services and registry-related operators",
        22: "Other administrative support staff",
        23: "Personal service workers",
        24: "Sellers",
        25: "Personal care workers and the like",
        26: "Skilled construction workers and the like, except electricians",
        27: "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
        28: "Workers in food processing, woodworking, clothing and other industries and crafts",
        29: "Cleaning workers",
        30: "Unskilled workers in agriculture, animal production, fisheries and forestry",
        31: "Unskilled workers in extractive industry, construction, manufacturing and transport",
        32: "Meal preparation assistants"
    }

    resultDataFrame["Mother's occupation"] = resultDataFrame["Mother's occupation"].map(occupation_dict_mother)
    resultDataFrame["Mother's occupation"] = resultDataFrame["Mother's occupation"].astype("category")

    occupation_dict_father = {
        1: "Student",
        2: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
        3: "Specialists in Intellectual and Scientific Activities",
        4: "Intermediate Level Technicians and Professions",
        5: "Administrative staff",
        6: "Personal Services, Security and Safety Workers and Sellers",
        7: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
        8: "Skilled Workers in Industry, Construction and Craftsmen",
        9: "Installation and Machine Operators and Assembly Workers",
        10: "Unskilled Workers",
        11: "Armed Forces Professions",
        12: "Other Situation",
        13: "(blank)",
        14: "Armed Forces Officers",
        15: "Armed Forces Sergeants",
        16: "Other Armed Forces personnel",
        17: "Directors of administrative and commercial services",
        18: "Hotel, catering, trade and other services directors",
        19: "Specialists in the physical sciences, mathematics, engineering and related techniques",
        20: "Health professionals",
        21: "Teachers",
        22: "Specialists in finance, accounting, administrative organization, public and commercial relations",
        23: "Intermediate level science and engineering technicians and professions",
        24: "Technicians and professionals, of intermediate level of health",
        25: "Intermediate level technicians from legal, social, sports, cultural and similar services",
        26: "Information and communication technology technicians",
        27: "Office workers, secretaries in general and data processing operators",
        28: "Data, accounting, statistical, financial services and registry-related operators",
        29: "Other administrative support staff",
        30: "Personal service workers",
        31: "Sellers",
        32: "Personal care workers and the like",
        33: "Protection and security services personnel",
        34: "Market-oriented farmers and skilled agricultural and animal production workers",
        35: "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
        36: "Skilled construction workers and the like, except electricians",
        37: "Skilled workers in metallurgy, metalworking and similar",
        38: "Skilled workers in electricity and electronics",
        39: "Workers in food processing, woodworking, clothing and other industries and crafts",
        40: "Fixed plant and machine operators",
        41: "Assembly workers",
        42: "Vehicle drivers and mobile equipment operators",
        43: "Unskilled workers in agriculture, animal production, fisheries and forestry",
        44: "Unskilled workers in extractive industry, construction, manufacturing and transport",
        45: "Meal preparation assistants",
        46: "Street vendors (except food) and street service providers"
    }

    resultDataFrame["Father's occupation"] = resultDataFrame["Father's occupation"].map(occupation_dict_father).astype("category")
    # resultDataFrame["Father's occupation"] = resultDataFrame["Father's occupation"]

    resultDataFrame["Displaced"] = resultDataFrame["Displaced"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Educational special needs"] = resultDataFrame["Educational special needs"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Debtor"] = resultDataFrame["Debtor"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Scholarship holder"] = resultDataFrame["Scholarship holder"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Gender"] = resultDataFrame["Gender"].map({1:"Male",0:"Female"}).astype("category")
    resultDataFrame["International"] = resultDataFrame["International"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Application order"] = resultDataFrame["Application order"].astype("category")
    resultDataFrame["Tuition fees up to date"] = resultDataFrame["Tuition fees up to date"].map({1:"Yes",0:"No"}).astype("category")
    resultDataFrame["Application order"] = resultDataFrame["Application order"].astype("category")
    resultDataFrame['Age group'] = pd.cut(resultDataFrame['Age at enrollment'], 
                            bins=[0, 20, 25, 30, 40, 100], 
                            labels=['<20', '20-25', '25-30', '30-40', '>40'])
    return resultDataFrame



if __name__ == "__main__":
    if not os.path.exists(".\\data"):
        os.makedirs(".\\data")
    if not os.path.exists(".\\data\\dataset.csv"):
        path = kagglehub.dataset_download(handle="naveenkumar20bps1137/predict-students-dropout-and-academic-success",path="dataset.csv")
        shutil.move(path, ".\\data")

    dataf: pd.DataFrame = pd.read_csv("data\\dataset.csv")
    dataf = changeFromNumericalToCategorical(dataf)
    dataf.to_csv("data\\datasetClean.csv",index=False)
    print("Data loaded and transformed")





