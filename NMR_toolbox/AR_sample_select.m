function[train_data, train_label, test_data, test_label] = AR_sample_select(sample_data, sample_label, occlusion_type, EachClassNum, ClassNum)
%%读取每张脸的26张照片，读取

temp = zeros(1,EachClassNum);
temp1 = zeros(1,EachClassNum);
temp([1:3,18:20])=1;
% temp([1])=1;


switch occlusion_type
    case 'sunglasses'        
        temp1([8:10,21:23])=1;        
    case 'scarf'       
        temp1([11:13,24:26])=1;
end
   
train_ind = logical(repmat(temp,1,ClassNum));
test_ind = logical(repmat(temp1,1,ClassNum));
    
train_data = sample_data(:,train_ind);
train_label = sample_label(:,train_ind);
    
test_data = sample_data(:,test_ind);
test_label = sample_label(:,test_ind);

end