{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# URL\n",
    "https://catalog.data.gov/dataset/behavioral-risk-factor-surveillance-system-brfss-national-cardiovascular-disease-surveilla"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                        RowId  YearStart LocationAbbr  \\\n",
      "0  BRFSS~2011~01~BR001~OVR01~Age-Standardized       2011           AL   \n",
      "1             BRFSS~2011~01~BR001~OVR01~Crude       2011           AL   \n",
      "2             BRFSS~2011~01~BR001~GEN01~Crude       2011           AL   \n",
      "3  BRFSS~2011~01~BR001~GEN01~Age-Standardized       2011           AL   \n",
      "4  BRFSS~2011~01~BR001~GEN02~Age-Standardized       2011           AL   \n",
      "\n",
      "  LocationDesc DataSource PriorityArea1  PriorityArea2 PriorityArea3  \\\n",
      "0      Alabama      BRFSS           NaN            NaN           NaN   \n",
      "1      Alabama      BRFSS           NaN            NaN           NaN   \n",
      "2      Alabama      BRFSS           NaN            NaN           NaN   \n",
      "3      Alabama      BRFSS           NaN            NaN           NaN   \n",
      "4      Alabama      BRFSS           NaN            NaN           NaN   \n",
      "\n",
      "   PriorityArea4                    Class  ... Break_Out_Category Break_Out  \\\n",
      "0            NaN  Cardiovascular Diseases  ...            Overall   Overall   \n",
      "1            NaN  Cardiovascular Diseases  ...            Overall   Overall   \n",
      "2            NaN  Cardiovascular Diseases  ...             Gender      Male   \n",
      "3            NaN  Cardiovascular Diseases  ...             Gender      Male   \n",
      "4            NaN  Cardiovascular Diseases  ...             Gender    Female   \n",
      "\n",
      "  ClassId TopicId  QuestionId  Data_Value_TypeID BreakOutCategoryId  \\\n",
      "0      C1      T1       BR001            AgeStdz              BOC01   \n",
      "1      C1      T1       BR001              Crude              BOC01   \n",
      "2      C1      T1       BR001              Crude              BOC02   \n",
      "3      C1      T1       BR001            AgeStdz              BOC02   \n",
      "4      C1      T1       BR001            AgeStdz              BOC02   \n",
      "\n",
      "  BreakOutId  LocationId                                   Geolocation  \n",
      "0      OVR01           1  POINT (-86.63186076199969 32.84057112200048)  \n",
      "1      OVR01           1  POINT (-86.63186076199969 32.84057112200048)  \n",
      "2      GEN01           1  POINT (-86.63186076199969 32.84057112200048)  \n",
      "3      GEN01           1  POINT (-86.63186076199969 32.84057112200048)  \n",
      "4      GEN02           1  POINT (-86.63186076199969 32.84057112200048)  \n",
      "\n",
      "[5 rows x 30 columns]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\bmcki\\AppData\\Local\\Temp\\ipykernel_30160\\4009407940.py:7: DtypeWarning: Columns (5) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  heart_disease_data = pd.read_csv(file_path)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Assuming the file path is '/mnt/data/Behavioral_Risk_Factor_Surveillance_System__BRFSS__-__National_Cardiovascular_Disease_Surveillance_Data.csv'\n",
    "file_path = 'Behavioral_Risk_Factor_Surveillance_System__BRFSS__-__National_Cardiovascular_Disease_Surveillance_Data.csv'\n",
    "\n",
    "# Reading the CSV file into a DataFrame\n",
    "heart_disease_data = pd.read_csv(file_path)\n",
    "\n",
    "# Display the first few rows of the dataframe\n",
    "print(heart_disease_data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(160160, 30)"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the rows and columns\n",
    "heart_disease_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['RowId', 'YearStart', 'LocationAbbr', 'LocationDesc', 'DataSource',\n",
       "       'PriorityArea1', 'PriorityArea2', 'PriorityArea3', 'PriorityArea4',\n",
       "       'Class', 'Topic', 'Question', 'Data_Value_Type', 'Data_Value_Unit',\n",
       "       'Data_Value', 'Data_Value_Alt', 'Data_Value_Footnote_Symbol',\n",
       "       'Data_Value_Footnote', 'Low_Confidence_Limit', 'High_Confidence_Limit',\n",
       "       'Break_Out_Category', 'Break_Out', 'ClassId', 'TopicId', 'QuestionId',\n",
       "       'Data_Value_TypeID', 'BreakOutCategoryId', 'BreakOutId', 'LocationId',\n",
       "       'Geolocation'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart_disease_data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowId                              0\n",
       "YearStart                          0\n",
       "LocationAbbr                       0\n",
       "LocationDesc                       0\n",
       "DataSource                         0\n",
       "PriorityArea1                 137280\n",
       "PriorityArea2                 160160\n",
       "PriorityArea3                  80080\n",
       "PriorityArea4                 160160\n",
       "Class                              0\n",
       "Topic                              0\n",
       "Question                           0\n",
       "Data_Value_Type                    0\n",
       "Data_Value_Unit                    0\n",
       "Data_Value                     81923\n",
       "Data_Value_Alt                     0\n",
       "Data_Value_Footnote_Symbol     78237\n",
       "Data_Value_Footnote            78237\n",
       "Low_Confidence_Limit           83847\n",
       "High_Confidence_Limit          83847\n",
       "Break_Out_Category                 0\n",
       "Break_Out                          0\n",
       "ClassId                            0\n",
       "TopicId                            0\n",
       "QuestionId                         0\n",
       "Data_Value_TypeID                  0\n",
       "BreakOutCategoryId                 0\n",
       "BreakOutId                         0\n",
       "LocationId                         0\n",
       "Geolocation                     3080\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "heart_disease_data.isna().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowId                 0\n",
       "YearStart             0\n",
       "LocationAbbr          0\n",
       "LocationDesc          0\n",
       "DataSource            0\n",
       "Class                 0\n",
       "Topic                 0\n",
       "Question              0\n",
       "Data_Value_Type       0\n",
       "Data_Value_Unit       0\n",
       "Data_Value_Alt        0\n",
       "Break_Out_Category    0\n",
       "Break_Out             0\n",
       "ClassId               0\n",
       "TopicId               0\n",
       "QuestionId            0\n",
       "Data_Value_TypeID     0\n",
       "BreakOutCategoryId    0\n",
       "BreakOutId            0\n",
       "LocationId            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Drop columns with any missing values based on the provided summary\n",
    "columns_to_drop_due_to_missing = [\n",
    "    'PriorityArea1', 'PriorityArea2', 'PriorityArea3', 'PriorityArea4',\n",
    "    'Data_Value', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',\n",
    "    'Low_Confidence_Limit', 'High_Confidence_Limit', 'Geolocation'\n",
    "]\n",
    "\n",
    "heart_disease_data = heart_disease_data.drop(columns=columns_to_drop_due_to_missing)\n",
    "\n",
    "# Display summary to verify changes\n",
    "heart_disease_data.isna().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(160160, 20)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check updated rows and columns to ensure no loss data\n",
    "heart_disease_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['RowId', 'YearStart', 'LocationAbbr', 'LocationDesc', 'DataSource',\n",
       "       'Class', 'Topic', 'Question', 'Data_Value_Type', 'Data_Value_Unit',\n",
       "       'Data_Value_Alt', 'Break_Out_Category', 'Break_Out', 'ClassId',\n",
       "       'TopicId', 'QuestionId', 'Data_Value_TypeID', 'BreakOutCategoryId',\n",
       "       'BreakOutId', 'LocationId'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check existing columns\n",
    "heart_disease_data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "160160"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check for unique ids\n",
    "heart_disease_data['RowId'].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'There are 160160 unique IDs in the dataset.'"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Number of unique ids\n",
    "unique_ids_count = heart_disease_data['RowId'].nunique()\n",
    "f\"There are {unique_ids_count} unique IDs in the dataset.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "YearStart\n",
       "2011    16016\n",
       "2012    16016\n",
       "2013    16016\n",
       "2014    16016\n",
       "2015    16016\n",
       "2016    16016\n",
       "2017    16016\n",
       "2018    16016\n",
       "2019    16016\n",
       "2020    16016\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Years of frequency\n",
    "year_frequency = heart_disease_data['YearStart'].value_counts().sort_values(ascending=False)\n",
    "year_frequency"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## There are 10 years with 16016 entries which makes the dataset consistant."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LocationAbbr\n",
       "AL     3080\n",
       "AK     3080\n",
       "ID     3080\n",
       "AZ     3080\n",
       "AR     3080\n",
       "CA     3080\n",
       "CO     3080\n",
       "CT     3080\n",
       "DE     3080\n",
       "DC     3080\n",
       "FL     3080\n",
       "GA     3080\n",
       "HI     3080\n",
       "IL     3080\n",
       "MS     3080\n",
       "IN     3080\n",
       "IA     3080\n",
       "KS     3080\n",
       "KY     3080\n",
       "LA     3080\n",
       "ME     3080\n",
       "MD     3080\n",
       "MA     3080\n",
       "MI     3080\n",
       "MN     3080\n",
       "MO     3080\n",
       "MT     3080\n",
       "NE     3080\n",
       "PA     3080\n",
       "NV     3080\n",
       "NH     3080\n",
       "NJ     3080\n",
       "NM     3080\n",
       "NY     3080\n",
       "NC     3080\n",
       "ND     3080\n",
       "OH     3080\n",
       "OK     3080\n",
       "OR     3080\n",
       "RI     3080\n",
       "WY     3080\n",
       "SC     3080\n",
       "SD     3080\n",
       "TN     3080\n",
       "TX     3080\n",
       "UT     3080\n",
       "VT     3080\n",
       "VA     3080\n",
       "WA     3080\n",
       "WV     3080\n",
       "WI     3080\n",
       "USM    3080\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Locations abbreviations\n",
    "loc_abbr = heart_disease_data['LocationAbbr'].value_counts().sort_values(ascending=False)\n",
    "loc_abbr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "52"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for 'LocationAbbr' column\n",
    "unique_loc_abbr_count = heart_disease_data['LocationAbbr'].nunique()\n",
    "unique_loc_abbr_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "160160"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Assuming each location abbreviation (LocationAbbr) has 3080 entries\n",
    "# and that there are 52 unique location abbreviations\n",
    "\n",
    "# Number of entries per location abbreviation\n",
    "entries_per_location = 3080\n",
    "\n",
    "# Total number of unique location abbreviations\n",
    "unique_locations = 52\n",
    "\n",
    "# Total number of entries\n",
    "total_entries = entries_per_location * unique_locations\n",
    "\n",
    "total_entries"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## There are a total of 52 states with 3080 per location making the data consistant."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for DataSource column\n",
    "heart_disease_data['DataSource'].nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Only one unique datasource: BRFSS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Class\n",
       "Risk Factors               102960\n",
       "Cardiovascular Diseases     57200\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Class column\n",
    "heart_disease_data['Class'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Risk Factors are higher which make outside influences and models to look into deeper"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Topic\n",
       "Acute Myocardial Infarction (Heart Attack)    22880\n",
       "Cholesterol Abnormalities                     22880\n",
       "Hypertension                                  22880\n",
       "Major Cardiovascular Disease                  11440\n",
       "Coronary Heart Disease                        11440\n",
       "Stroke                                        11440\n",
       "Diabetes                                      11440\n",
       "Nutrition                                     11440\n",
       "Obesity                                       11440\n",
       "Physical Inactivity                           11440\n",
       "Smoking                                       11440\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Topic column\n",
    "heart_disease_data['Topic'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Three columns(Acute Myocardial Infarction (Heart Attack), Cholesterol Abnormalities, and Hypertension) have 22880 values at the highest, and the remaning eight values have the lowers of 11440 values. Can create different graphs with models to see the uniqueness in each. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Question\n",
       "Prevalence of major cardiovascular disease among US adults (18+); BRFSS                                  11440\n",
       "Prevalence of acute myocardial infarction (heart attack) among US adults (18+); BRFSS                    11440\n",
       "Prevalence of post-hospitalization rehabilitation among heart attack patients, US adults (18+); BRFSS    11440\n",
       "Prevalence of coronary heart disease among US adults (18+); BRFSS                                        11440\n",
       "Prevalence of stroke among US adults (18+); BRFSS                                                        11440\n",
       "Prevalence of cholesterol screening in the past 5 years among US adults (20+); BRFSS                     11440\n",
       "Prevalence of high total cholesterol among US adults (20+); BRFSS                                        11440\n",
       "Prevalence of diabetes among US adults (18+); BRFSS                                                      11440\n",
       "Prevalence of consuming fruits and vegetables less than 5 times per day among US adults (18+); BRFSS     11440\n",
       "Prevalence of obesity among US adults (20+); BRFSS                                                       11440\n",
       "Prevalence of physical inactivity among US adults (18+); BRFSS                                           11440\n",
       "Prevalence of current smoking among US adults (18+); BRFSS                                               11440\n",
       "Prevalence of hypertension among US adults (18+); BRFSS                                                  11440\n",
       "Prevalence of hypertension medication use among US adults (18+) with hypertension; BRFSS                 11440\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Question column\n",
    "heart_disease_data['Question'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# All questions have the same values count of 11440"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Data_Value_Type\n",
       "Crude               101920\n",
       "Age-Standardized     58240\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Data_Value_Type column\n",
    "heart_disease_data['Data_Value_Type'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## The crude death rate does not reflect changes in the age structure of the population over time. Age-standardisation is a method of adjustment to allow for the effect of variation in the population age structure when comparing death rates for different years or different locations. Great to look at different models to check the influences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Data_Value_Unit\n",
       "Percent (%)    160160\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Data_Value_Unit column\n",
    "heart_disease_data['Data_Value_Unit'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# All data values go by a percentage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Data_Value_Alt\n",
       "-1.0     59500\n",
       "-2.0     22423\n",
       " 3.0       504\n",
       " 2.9       502\n",
       " 3.3       493\n",
       "         ...  \n",
       " 48.3        6\n",
       " 99.3        5\n",
       " 99.1        4\n",
       " 0.1         2\n",
       " 99.4        1\n",
       "Name: count, Length: 996, dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Data_Value_Alt column\n",
    "heart_disease_data['Data_Value_Alt'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Will check back later to check on how the correlation impacts the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Break_Out_Category\n",
       "Race       72800\n",
       "Age        43680\n",
       "Gender     29120\n",
       "Overall    14560\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Break_Out_Category column\n",
    "heart_disease_data['Break_Out_Category'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# There are different values in each of the 4 categories. Race is impacted the most, while all three combined (Race, Age, Gender), are the lowest. Definitely a great factor to see why there are more values individually that grouped."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Break_Out\n",
       "Overall               14560\n",
       "Male                  14560\n",
       "Female                14560\n",
       "Non-Hispanic White    14560\n",
       "Non-Hispanic Black    14560\n",
       "Non-Hispanic Asian    14560\n",
       "Hispanic              14560\n",
       "Other                 14560\n",
       "25-44                  7280\n",
       "45-64                  7280\n",
       "65+                    7280\n",
       "35+                    7280\n",
       "75+                    7280\n",
       "18-24                  5720\n",
       "20-24                  1560\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Unique values for Break_Out column\n",
    "heart_disease_data['Break_Out'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Different values for all age groups and races. Would be interesting to look at both groups individually and groups to see the impact as shown in the 'Break_Out_Category' column"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check back on columns with IDs:\n",
    "\"ClassId\", \"TopicId\", \"QuestionId\", \"Data_Value_TypeID\", \"BreakOutCategoryId\", \"BreakOutId\", and \"LocationId\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "heart_disease_data.to_excel('heart_disease_data_cleaned.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Export cleaned up data"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
