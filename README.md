### ToBeeOrNotToBee

1. 建立虛擬環境 (For window)
    
    ```jsx
    python -m venv venv
    .\venv\Scripts\activate
    ```
    
2. 開啟 mongodb sever
    
    
    
3. 安裝依賴包、建立mongodb資料庫
    
    ```jsx
    pip install pandas
    pip install pymongo
    pip install openpyxl
    python .\DBSoundBasicInfo.py
    ```
    
4. 安裝依賴包，確定機器學習預測的程式能運行 (儲存進mongodb的過程被我註解掉了)
    
    ```jsx
    pip install scipy
    pip install librosa
    python .\Beepre_mongo.py
    ```
    
5. 進到kedro
    
    ```jsx
    cd .\audio-classification\
    pip install kedro
    # 在 ToBeeOrNotTobee\audio-classification\conf\base\parameters.yml
    裡面更改knn_model_path 位置 
    #更改完畢以後
    kedro run
    ```
    

### bee_monitoring

1. 先連上網路
2. 開啟建立完成的mongodb
3. 安裝配件、開啟後端
    
    ```jsx
    	cd ../..
    	cd .\bee_monitoring_system\
    	pip install flask
    	pip install flask_cors
    ```
    
4. 開網頁
