#Config data
Dataset_Image = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Input_model/Images"               # Đường dẫn data set
Output_Json_Task_2_1 = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Task2.1_output"            # Đường dẫn ra output json task 2.1: Dùng YOLO OBB
Output_Json_Task_2 = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Task2_output"                # Đường dẫn ra output json task 2
Output_Json_Task_3 = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Task3_output"                # Đường dẫn ra output json task 3
Output_Json_Task_4 = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Task4_output"                # Đường dẫn ra output json task 4
Output_Excel_Task_4 = "E:/Hcmut material/Project_1/SubmissionFinalCode/DatasetPredict/Task4_output/result.csv"   #Đường dẫn ra output excel task 4

#Config model
model_path_layoutlmv3 = "E:/Hcmut material/Project_1/checkpoint-10000"    # Đường dẫn file checkpoint layoutlmv3
model_path_yolo = "E:/Hcmut material/Project_1/weights/best.pt"           # Đường dẫn file checkpoint yolo cho task 3 4 5
model_path_yolo_0 = "E:/Hcmut material/Project_1/weights_text/weights/best.pt"    # Đường dẫn file checkpoint yolo cho task 2.1

Task2_1Config = {
    "input": Dataset_Image,
    "ouput": Output_Json_Task_2_1,
    "weight": model_path_yolo_0
}

Task2Config = {
    "input_json": Output_Json_Task_2_1,
    "input_images": Dataset_Image,
    "output": Output_Json_Task_2
}

Task3Config = {
    "model_path": model_path_layoutlmv3, 
    "data_dir_images": Dataset_Image,
    "data_dir_json": Output_Json_Task_2,
    "labels": ["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL", "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
    "device": "cuda", 
    "output_dir": Output_Json_Task_3
}

Task4Config = {
    "input_images": Dataset_Image,
    "input_json": Output_Json_Task_3,
    "output_excel": Output_Excel_Task_4,
    "output_json": Output_Json_Task_4,
    "yolo_weight": model_path_yolo,
    "device": "cuda"
}

def returnTestTask2_1_Config():
    return Task2_1Config

def returnTestTask2_Config():
    return Task2Config

def returnTestTask3_Config():
    return Task3Config

def returnTestTask4_Config():
    return Task4Config