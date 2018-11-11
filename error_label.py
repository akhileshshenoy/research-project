from google.cloud import vision
from google.cloud.vision import types

vision_client = vision.ImageAnnotatorClient()
def calculate_error(path_org,path_new):
    
    # path_org = 'prg_image.jpg'
    # path_new = 'new_image.jpg'
    
    # getting labels of path_org.jpg and storing it in dictionary label_org
    with io.open(path_org,'rb') as image_org:
        content_org = image_org.read()

    image_org = types.Image(content_org=content_org)

    response_org = vision_client.label_detection(image_org=image_org)
    labels_org_l = response_org.label_annotations
    
    label_org = {}
    
    for label in labels_org_l:
        #print(label.description,label.score)
        label_org.update({label.description:label.score})
    
    # getting labels of path_new.jpg and storing it in dictionary labl_new    
    with io.open(path_new,'rb') as image_new:
        content_new = image_new.read()

    image_new = types.Image(content_new=content_new)

    response_new = vision_client.label_detection(image_new=image_new)
    labels_new_1 = response_new.label_annotations
    
    label_new = {}
    
    for label in labels_new_1:
        #print(label.description,label.score)
        label_new.update({label.description:label.score})
        
    #d4 is dictioary used for storing labels and confidences of label_new after preprocessing 
    #d3 is used for inside operations DO NOT CHANGE
    d3 = {}
    d4 = {}
    
    
    #adding common labels to d3
    for x in label_new:
        if(x in label_org):
            if(label_new[x]>=label_org[x]):
                label_new[x]=label_org[x]
            d3.update({x:label_new[x]})
    
    
    #setting labels in org but bot in new as 0 and adding to d3
    for x in label_org:
        if(not(x in d3)):
            d3.update({x:0})
    
    
    ### --------------IF YOU WANT TO ADD RANKING FACTOR ADD HERE ----------------- ###
    
    
    #keeping both dictionaries label_org(labels of org) and d4(labels of new) in same order
    for x in label_org:
        for y in d3:
            if(x==y):
                d4.update({x:d3[x]})
    
    
    #Creating numpy arrays of confidence values in d_1(org) and d_2(new)
    d_1 = []
    d_2 = []
    for x in label_org:
        d_1.append(label_org[x])
    for y in d4:
        d_2.append(d4[y])
    conf_1 = np.array(d_1)
    conf_2 = np.array(d_2)
    
    
    #calculating weights of each confidence label
    weight_1 = np.array((conf_1 / np.sum(conf_1)))
    weight_2 = np.array((conf_2/conf_1)*(weight_1))
    
    
    #ensuring weight_2 values are always lesser than weight_1 values
    #for idx,i in enumerate(weight_1):
     #   if(weight_2[idx]>weight_1[idx]):
      #      weight_2[idx]=weight_1[idx]
            
    
    #error calculation
    error = (1.0-np.sum(weight_2))*100
    return error
label_org = {"Statue" : 95.0,"Landmark" :90.0,"Palace" : 78.0,"Tourism":69.0,"Mansion":59.0}
label_new = {"Mansion":59.0,"Statue" : 90.0,"Landmark" :92.0,"Palace" : 60.0,"Sky":88.0,"Space":55.0}
e = error_calculate(label_org,label_new)
print(e)
    

