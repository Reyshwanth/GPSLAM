
import numpy as np
oarray=np.array([1,-4,5])

def sign_change_count(oarray):
    sign_change=0
    skip_index=[]
    for i in range(len(oarray)-1):
        j=1
        if i in skip_index:
            continue
        while(oarray[i]*oarray[i+j]==0):
            skip_index.append(i+j)
            j+=1
            
        if oarray[i]*oarray[i+j]<0:
            sign_change+=1 
        
    return sign_change

sign_change=sign_change_count(oarray)
#breakpoint()   
while(sign_change==2):
    n1array=np.append(oarray,0)
    n2array = np.insert(oarray,0,0)
    oarray = n1array    + n2array
    #breakpoint()
    sign_change=sign_change_count(oarray)
print(oarray)
