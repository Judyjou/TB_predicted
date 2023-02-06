
因此專案與實驗室合作仍有團隊尚在優化，因此暫不提供yaml環境檔與h5模型訓練檔


============== 由網頁執行判斷 ==============
***為避免執行網頁失效，因此須使 dl_in_prod\static\image 中無其餘圖檔，如有其他圖檔需手動刪除，
如有其他額外產生之圖片處理資料夾亦須手動刪除。ex: store_reOri,reSave,temp, processing 資料夾，以上資料夾中無其餘圖檔網頁可正常執行。 ***  

上傳之圖片需再 dl_in_prod\static\image 此資料夾中才可正確執行判斷。

1.app.py主要連結html,py執行檔，開啟spyder 執行app.py
2.至chrome 127.0.0.1:8000 即開啟網頁
3.上傳照片附檔名須為png, jpg


============== 另單純執行py檔則為 ==============
predictFunction.py
此檔為負責判斷影像之程式
1.須將圖先手動放置 static\image資料夾中(此處副檔名可為dcm,png, jpg之檔案)
2.predictFunction.py 執行
結果將顯示於console中且於 dl_in_prod 資料夾中會寫出 Result.txt

I.GoogLeNet_STTI_o2s8_exn10416Pre_SGD_SHZ_correct_5.h5 為主要引用之訓練模型
II.Tensorflow_Gpu.yaml 是於sever上執行的環境
III.extraPip.txt 是額外載的套件
IV.extraFunction 為predictFunction.py執行會需要用到的額外程式