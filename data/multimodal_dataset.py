import pandas as pd
from scipy.io import arff
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler, MaxAbsScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset
import re
import string
from transformers import BertTokenizer, AutoTokenizer,LongformerTokenizerFast, EncoderDecoderModel#,  LongformerForSeq2SeqLM
from dataset.report_parser import section_text
def process_csv_transformer(df):
    
    print("dffff",pd.DataFrame(df).shape,df.columns[-15:-1])
    df = df.drop(columns=['Unnamed: 0', 'index', 'subject_id', 'hadm_id', 'stay_id', 'intime',
       'outtime', 'note_id', 'note_type', 'note_seq', 'charttime',
       'storetime','admittime', 'dischtime', 'edregtime', 'edouttime'],axis=1)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = [col for col in df.columns if col not in categorical_columns]  
    
    print("nummmmm2",numerical_columns,categorical_columns)
    #train.dropna(axis = 0, how ='any',inplace=True) 
    return df, numerical_columns, categorical_columns

def process_csv_transformer(df):
    
    print("dffff",pd.DataFrame(df).shape,df.columns[-15:-1])
    df = df.drop(columns=['Unnamed: 0', 'index', 'subject_id', 'hadm_id', 'stay_id', 'intime',
       'outtime', 'note_id', 'note_type', 'note_seq', 'charttime','subject_id',
       'storetime','admittime', 'dischtime', 'edregtime', 'edouttime'],axis=1)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Add columns based on the number of unique values
    threshold = 3
    for col in df.columns:
        if df[col].nunique() < threshold and col not in categorical_columns:
            categorical_columns.append(col)
    numerical_columns = [col for col in df.columns if col not in categorical_columns]  
    
    print("nummmmm",numerical_columns,categorical_columns)
    #train.dropna(axis = 0, how ='any',inplace=True) 
    return df, numerical_columns, categorical_columns

# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

def remove_technique_section(report):
    """
    Remove the technique section from the radiology report.
    """
    # Define the regex pattern for the technique section
    # This pattern assumes that the section header is "Technique" followed by a colon or a new line,
    # and the section ends before the next section header (e.g., "Findings", "Impression", etc.)
    # print("rawww report**************************************************",report)
    # pattern = r'TECHNIQUE:(.*?)(?=\n[A-Z][a-z]*:|\Z)'#r'TECHNIQUE:(.*?)(?=\n[A-Z ]+:|\Z)'
    # pattern =  r'"[A-Z0-9][A-Z0-9. ]*:"'
    # print("patttternnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",pattern)
    # # Use re.sub() to remove the technique section
    # cleaned_report = re.sub(pattern, '', report, flags=re.DOTALL)

    # pattern2 = r'PROCEDURE:(.*?)(?=\n[A-Z ]+:|\Z)'
    # cleaned_report = re.sub(pattern, '', report, flags=re.DOTALL)

    # pattern3 = r'Detailed TECHNIQUE:(.*?)(?=\n[A-Z ]+:|\Z)'
    # cleaned_report = re.sub(pattern3, '', report, flags=re.DOTALL)
    # pattern4 = r'DOSE:(.*?)(?=\n[A-Z ]+:|\Z)'
    # cleaned_report = re.sub(pattern4, '', report, flags=re.DOTALL)
    # print("cleaned_report*******************************",cleaned_report)
    # pattern =r"[A-Z0-9][A-Z0-9. ]*"
    pattern = r"[A-Z0-9][A-Z0-9. ]*:"
    # pattern = r"[A-Z0-9]*:"
    headings = list(re.finditer(pattern, report))

    # Initialize the new text
    # pattern = r"[A-Z0-9][A-Z0-9. ]*:"
    # print("nowwwwwwwwwwwwwwwwwww",headings)
    new_text = report

    # Find the TECHNIQUE heading
    exclusion_list = ["TECHNIQUE:","DETAILED TECHNIQUE:","DOSE:","PROCEDURE:","FLUOROSCOPY TIME AND DOSE:","PROCEDURE DETAILS:","FLUOROSCOPY TIME AND DOSE:","OPERATORS:","NOTIFICATION:","COMPARISON:","EXAMINATION:","DLP:"]#,"INDICATION:"] 
    # summarize 
    for i in range(len(headings)):
        # print("grouppppp",headings[i].group())
        # print("salaaaaaaaaaaaaaaaaaammmmmmmm",headings[i])
        if headings[i].group() in exclusion_list:
            # print("salaammmmmmmmmmm","yessssssssssssssssss",headings[i].group())
            # start = headings[i].start()
            start = new_text.find(headings[i].group())
            # Find the next heading if it exists
            # print("star2222222222222",start)
            if i + 1 < len(headings):
                # end = headings[i + 1].start()
                end = new_text.find(headings[i+1].group())
            else:
                end = len(new_text)
            # print("headinggggggggggggg222222222",headings[i+1].group())
            # print("enddddddddddddd",end)
            # Remove text from TECHNIQUE to the next heading
            # if headings[i].group()=="TECHNIQUE:":
            #     print("techniquuuee22222",report[:start] + report[end:])
            new_text = new_text[:start] + new_text[end:]
        

    # print(new_text)
    return new_text



def remove_instruction_info(text):
    # Define a regex pattern to match the name and DOB lines
    # The pattern looks for "Name:" followed by a name, and "DOB:" followed by a date
    text = re.sub(r"Name:.*?Sex:.*?\n", "", text, flags=re.DOTALL)
    pattern = r"Name:.*\nUnit No:.*\nAdmission Date:.*\nDischarge Date:.*\nDate of Birth:.*\nSex:.*\nService:.*\n|Discharge Instructions:.*$"
    
    # Clean the summaries by applying the regex to remove patient info
    clean_text = re.sub(pattern, "", text, flags=re.DOTALL) 
    return clean_text


def process_text(text):
    # Convert text to lowercase
    # print("INNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",text)
    # print("input",text)
    text = remove_technique_section(text)
    text = remove_instruction_info(text)
    # print("output"#,text)
    # text = section_text(text)
    # print("OUTTTT**********************************************",text)
    
    
    # Remove all dates (considering common date formats)
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # e.g., 12/31/1999 or 31-12-1999
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',    # e.g., 1999/12/31 or 1999-12-31
        r'\b\d{1,2} \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \b\d{2,4}',  # e.g., 31 Dec 1999
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \b\d{1,2}, \b\d{2,4}'  # e.g., Dec 31, 1999
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, '', text)
    text = text.lower()
    # Remove all numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove all punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))
    text= re.sub(r'\s+', ' ', text)
    text = text.strip()
    # text = text.strip()
    # text = text.replace(r'\s+', ' ', regex=True)
    text = re.sub(r'\b' + re.escape("cm") + r'\b', '', text)
    text = re.sub(r'\b' + re.escape("mm") + r'\b', '', text) 
    text = re.sub(r'\b' + re.escape("year old") + r'\b', '', text)
    # stop_words = stopwords.words('english')
    # # text =  ''.join([word for word in text if word not in stop_words])
    # for word in stop_words:
    #     if word in text:
    #         text = text.replace(f' {word} ', ' ')
    return text

class multimodaldataset2(Dataset):
    def __init__(self, tabular_df,numerical_columns, categorical_columns):
        self.df = tabular_df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        print("tabular-df",tabular_df.shape)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        # self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalGPT-base-zh")
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        
        report = self.df.loc[index,"text"]
        # print("raw report",report)
        EHR = self.df.drop("text",axis=1)#.iloc[index]

        # _, numerical_columns, categorical_columns = process_csv_transformer(EHR)
        data = {}
        
        for nc in self.numerical_columns:
            
            x = torch.tensor(EHR[nc][index],
                             dtype=torch.float32)
            x = torch.unsqueeze(x, dim=0)
            data[nc] = x
        for cc in self.categorical_columns:
            x = torch.tensor(EHR[cc][index],
                             dtype=torch.int32)
            x = torch.unsqueeze(x, dim=0)
            data[cc] = x
            
        # label = self.df.loc[index,"readmitted_binarized"]
        # print("vlll",report)
        clean_report = process_text(report)
        # print("clean report",clean_report)
        # print("reporttttt",clean_report)
        
        # tokenized_report = self.tokenizer.encode_plus(
            
            
        #     clean_report,
        #     add_special_tokens=True,
        #     max_length=879,
        #     # pad_to_max_length=True,
        #     padding='max_length',
        #     return_token_type_ids=True,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        #     )
        # print("vlll",clean_report)
        
        # tokenized_report=self.tokenizer.encode_plus(clean_report,return_tensors="pt",padding="max_length", max_length = 1459) 
        # tokenized_report = self.tokenizer(clean_report, truncation=True, padding=True, max_length=1000)
        # print("tookenn",tokenized_report)


        # summarizer_model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
        # input_ids = self.tokenizer(clean_report, return_tensors="pt").input_ids
        # output_ids = summarizer_model.generate(input_ids)

        # Get the summary from the output tokens
        # summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print("pre_summmaryyyyyy", clean_report)
        # print("summmaryyyyyy", summary)
        # print("FINALLLLYYY")
        tokenized_report=self.tokenizer.encode_plus(clean_report,return_tensors="pt",padding="max_length", max_length = 3000,truncation=True)
        # summary_ids = summarizer_model.generate(clean_report, max_length=800, min_length=500, length_penalty=2.0, num_beams=4, early_stopping=True)
        # summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print("summaryyyyy",summary)

        return tokenized_report,data
    



class multimodaldataset(Dataset):
    def __init__(self, tabular_df,numerical_columns, categorical_columns,num_embedding_list,text):
        self.df = tabular_df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
       
        self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
        self.num_embedding_list = num_embedding_list
        self.text = text
        # EHR = self.df

        data = []



        self.numerical_data = numerical_columns
        self.categorical_data = categorical_columns
        # self.numerical_data = []
        # self.categorical_data = []
        
        # for i in range(len(self.df)):
        #     EHR = self.df.iloc[i]
        #     numerical_row = []
        #     categorical_row = []
        #     for nc in self.numerical_columns:
                
        #         # print("nccccccccc",torch.tensor([EHR[nc]]).shape)
        #         numerical_row.append(torch.tensor([EHR[nc]], dtype=torch.float32).unsqueeze(1))
            
            
            
        #     for cc in self.categorical_columns:
                
        #         categorical_row.append(torch.tensor([EHR[cc]], dtype=torch.float32).unsqueeze(1))

        #     self.numerical_data.append(torch.cat(numerical_row, dim=1))
        #     self.categorical_data.append(torch.cat(categorical_row, dim=1))



        
            # print("lennnnnn",len(self.numerical_data),self.numerical_data[0].shape)
       

        
        # data = torch.cat(self.numerical_row)    
        # Concatenate all data along the feature dimension
        
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        # self.numerical_data = self.numerical_data.unsqueeze(0)
        # self.categorical_data = self.categorical_data.unsqueeze(0)
        # print(self.numerical_data[index].shape)  # Shape of numerical data tensor
        # print(self.categorical_data[index].shape)
        data = torch.cat([self.numerical_data[index], self.categorical_data[index]], dim=0)
        report = self.text.iloc[index]#self.df.loc[index,"text"]
        
        # EHR = self.df#self.df.drop("text",axis=1)

        # data = []
        # numerical_data = []
        # categorical_data = []
        
        
        # for nc in self.numerical_columns:
            
            
        #     x = torch.tensor([EHR[nc][index]],
        #                      dtype=torch.float32)
            
        #     x = torch.unsqueeze(x, dim=1)
        #     numerical_data.append(x)
            
        
        # numerical_data = torch.cat(numerical_data, dim=1)
        
        # data.append(numerical_data)
        # for cc in self.categorical_columns:
        #     x = torch.tensor([EHR[cc][index]],
        #                      dtype=torch.int32)
        #     x = torch.unsqueeze(x, dim=1)
            
        #     categorical_data.append(x)
            
        
        # categorical_data = torch.cat(categorical_data, dim=1)

        # data.append(categorical_data)

        # # Concatenate all data along the feature dimension
        # data = torch.cat(data, dim=1)  
        
        # clean_report = process_text(report)
        
        
        tokenized_report=self.tokenizer(report, add_special_tokens=True,return_tensors="pt",padding="max_length",max_length = 4096,truncation=True)#, padding="max_length",truncation=True,max_length = 512) #truncation=True,
        # tokens = self.tokenizer.tokenize(report)

# Print the tokens
        # print(tokens)
        
        # print("HHHHHHHHHHHh",data.shape)
        return tokenized_report,data,torch.tensor(self.num_embedding_list)
    

# class multimodaldataset(Dataset):
    # def __init__(self, tabular_df,numerical_columns, categorical_columns):
    #     self.df = tabular_df
    #     self.numerical_columns = numerical_columns
    #     self.categorical_columns = categorical_columns
        
    #     print("tabular-df",tabular_df.shape)
    #     # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     self.tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer")
    #     # self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalGPT-base-zh")
        
    # def __len__(self):
    #     return len(self.df)
    # def __getitem__(self, index):
        
    #     report = self.df.loc[index,"text"]
    #     # print("raw report",report)
    #     EHR = self.df.drop("text",axis=1)#.iloc[index]

    #     # _, numerical_columns, categorical_columns = process_csv_transformer(EHR)
    #     data = {}
        
    #     for nc in self.numerical_columns:
            
    #         x = torch.tensor(EHR[nc][index],
    #                          dtype=torch.float32)
    #         x = torch.unsqueeze(x, dim=0)
    #         data[nc] = x
    #     for cc in self.categorical_columns:
    #         x = torch.tensor(EHR[cc][index],
    #                          dtype=torch.int32)
    #         x = torch.unsqueeze(x, dim=0)
    #         data[cc] = x
            
    #     # label = self.df.loc[index,"readmitted_binarized"]
    #     # print("vlll",report)
    #     clean_report = process_text(report)
    #     # print("clean report",clean_report)
    #     # print("reporttttt",clean_report)
        
    #     # tokenized_report = self.tokenizer.encode_plus(
            
            
    #     #     clean_report,
    #     #     add_special_tokens=True,
    #     #     max_length=879,
    #     #     # pad_to_max_length=True,
    #     #     padding='max_length',
    #     #     return_token_type_ids=True,
    #     #     return_attention_mask=True,
    #     #     return_tensors='pt'
    #     #     )
    #     # print("vlll",clean_report)
        
    #     # tokenized_report=self.tokenizer.encode_plus(clean_report,return_tensors="pt",padding="max_length", max_length = 1459) 
    #     # tokenized_report = self.tokenizer(clean_report, truncation=True, padding=True, max_length=1000)
    #     # print("tookenn",tokenized_report)


    #     # summarizer_model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
    #     # input_ids = self.tokenizer(clean_report, return_tensors="pt").input_ids
    #     # output_ids = summarizer_model.generate(input_ids)

    #     # Get the summary from the output tokens
    #     # summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     # print("pre_summmaryyyyyy", clean_report)
    #     # print("summmaryyyyyy", summary)
    #     # print("FINALLLLYYY")
    #     tokenized_report=self.tokenizer.encode_plus(clean_report,return_tensors="pt",padding="max_length", max_length = 3500)
    #     # summary_ids = summarizer_model.generate(clean_report, max_length=800, min_length=500, length_penalty=2.0, num_beams=4, early_stopping=True)
    #     # summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    #     # print("summaryyyyy",summary)

    #     return tokenized_report,data
