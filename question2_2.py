import pandas as pd


data1=pd.read_excel("D:\\2023数模国赛\\附件4_c.xlsx",sheet_name="Sheet2")
dict_loss={i:j for i,j in zip(data1['单品编码'], data1['损耗率(%)'])}
print(dict_loss+6)
