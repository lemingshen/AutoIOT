import os
import numpy as np

from typing import List
from langchain import hub
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.utilities import PythonREPL
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.google_search import GoogleSearchResults
from langchain.chains.combine_documents import create_stuff_documents_chain


programming_language = "Python"
os.environ["OPENAI_API_KEY"] = "your key"
os.environ["TAVILY_API_KEY"] = "your key"
session_configuration = {"configurable": {"session_id": "<AutoNLP>"}}
session_configuration1 = {"configurable": {"session_id": "<AutoNLP1>"}}
session_configuration2 = {"configurable": {"session_id": "<AutoNLP2>"}}

# modules and tools
llm = ChatOpenAI(model_name="gpt-4o")
embeddings = OpenAIEmbeddings()
search = TavilySearchResults()
message_history = ChatMessageHistory()
message_history1 = ChatMessageHistory()
message_history2 = ChatMessageHistory()
conversation = ConversationChain(llm=llm)
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
text_splitter = RecursiveCharacterTextSplitter()

# supvervised related settings
epoch_number = 3
# accuracy_threshold = 90

user_input_multimodal = """* Target *
Given a multimodal dataset containing three modalities: audio, depth camera, and radar data. Please write some Python code to implement a multimodal human activity recognition system. PyTorch is the preferred framework for machine learning.

* Workflow *
1. Load all the audio, depth camera, and radar data from the dataset.
2. Split the data into training and testing parts.
3. Construct and train a multimodal machine learning model using the training data.
4. Evaluate the activity recognition accuracy of the model using the testing data.
5. Output the average recognition accuracy.

* Input *
The file path to the multimodal dataset

* Output Format *
Average recognition accuracy on test data: 95%

* Remarks *
- There are four 'npy' files in the dataset path, i.e., 'audio.npy', 'depth.npy', 'radar.npy', and 'label.npy'. Each file contains 7484 data records.
- The shape of 'audio.npy' is (7484, 20, 87).
- The shape of 'depth.npy' is (7484, 16, 112, 112).
- The shape of 'radar.npy' is (7484, 20, 2, 16, 32, 16).
- The shape of 'label.npy' is (7484,).
- There are 11 activities in total, with label ranging from 0 to 10"""


expert_user_input = """* Target *
Given the 'XRF55' dataset, please write some Python code to implement a human activity recognition system. PyTorch is the preferred framework for machine learning.

* Workflow *
1. Load all the RFID data from the dataset.
2. Split the data into training and testing parts.
3. Construct and train a ResNet-18 model using the training data.
4. Evaluate the activity recognition accuracy of the model using the testing data.
5. Output the average recognition accuracy.

* Input *
The file path to the XRF55 dataset

* Output Format *
Average recognition accuracy on test data: 95%

* Remarks *
- The dataset is a single 'npy' file with a shape of (4400, 23, 148). 
- The corresponding label is a single 'npy' file with a length of 4400.
- The dataset contains 4400 RFID data records. Each data has a shape of (23, 148).
- There are 11 activities in total, with label ranging from 0 to 10"""


nonexpert_user_input = """* Target *
Given the 'XRF55' dataset, please write some Python code to:
1.	Load all the RFID data from the dataset
2.	Train a model that can classify all 55 actions with 80% of the data record
3.	Output the action detection results for the remaining 20% of the data record
4.	Calculate the accuracy of action detection results

* Input *
The file path to the 'XRF55' dataset: './RFID_project/dataset/Raw_dataset/'.

* Output Format *
Case {RFID data record name1, RFID data record name2, RFID data record name3, …}
Action result label: [0, 1, 2, …]
Action detection result: [0, 1, 2, ...]
Detection accuracy: 0.92

* Remarks *
- I only have a zip file (already extracted) named 'dataset.zip' downloaded from its official website.
- I have unzipped it and moved it to './RFID_project/dataset/Raw_dataset/.'
- The program should be directly executed in './RFID_project/'.
- Output the code execution requirements to  './RFID_project/requirement.txt'
"""

user_input_ECG = """* Target *
Given the 'MIT-BIH Arrhythmia Database', please first load all the ECG records and detect all the R-peaks. Then, evaulate the detection result and output the detection accuracy for each record. Please output the detection result and the detection accuracy following the output format stated below. The 'R-peak indices' must be a numpy list.

* Input *
The file path to the MIT-BIH Arrhythmia Database.

* Output Format *
Case {ECG data record name}
R-peak indices: [0, 1, 2, ...]
Detection accuracy: 0.92

Case {ECG data record name}
R-peak indices: [...]
Detection accuracy: 0.90


* Remarks *
- I only have a zip file (already extracted) named 'mit-bih-arrhythmia-database-1.0.0.zip' downloaded from its official website.
- I have extracted the file to a folder.
- The program should be directly executed."""

user_input_IMU = """* Target *
Given the WIreless Sensor Data Mining (WISDM) dataset containing Accelerometer data collected from smartphone, please develop a human activity recognition system. Please first preprocess the WISDM data and divide it into two parts for training and testing. Then, construct a machine learning model to recognize the activity. Finally, output the average recognition accuracy on the test dataset. PyTorch framework is preferred and I have a powerful GPU. The generated code should be directly executable.

* Remarks *
- I only have a gz file (already extracted) named 'WISDM_ar_latest.tar.gz' downloaded from its official website.
- I have extracted the file to a folder, containing the following files:
    - WISDM_ar_v1.1_raw.txt
    - WISDM_ar_v1.1_raw_about.txt
    - WISDM_ar_v1.1_trans_about.txt
    - readme.txt
    - WISDM_ar_v1.1_transformed.arff

* Input *
The path to the extracted folder.

* Output Format *
Average recognition accuracy: 0.90"""

user_input = """* Target *
Given the XRF55 dataset containing heatmaps extracted from mmWave radars that capture different human-computer interaction motions, please develop a human motion recognition system. Please first load and preprocess the XRF55 dataset and divide it into two parts for trianing and testing. Then, construct a powerful machine learning model to recognize the activity. Finally, output the average recognition accuracy on the test dataset. PyTorch framework is preferred and I have a powerful GPU. The generated code should be executable.

* Remarks *
- I have downloaded the dataset to a folder, containing 4 sub-folders named from 1 to 4. Each sub-folder contains a lot of npy files.
- The name of each npy file is formated as '<user_id>_<activity_id>_<trial_number>.npy', for example, 01_01_02.npy means the second data sample collected from the first user performing the first activity.
- Each npy array has a shape of (1, 17, 256, 128).
- I only want to train a model to classify the 'Human-Computer Interaction' group, which contains 11 activities.

* Input *
The path to the folder.

* Output Format *
Average recognition accuracy: 0.92"""
