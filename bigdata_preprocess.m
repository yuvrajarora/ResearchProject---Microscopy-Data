%% Append various day data to a single mat file

for i=1:183948
data(i).intensG = rem_pbleach(data(i).intensG,50) - mean(rem_pbleach(data(i).intensG,50));
data(i).intensR = rem_pbleach(data(i).intensR,50) - mean(rem_pbleach(data(i).intensR,50));
end
data1 = struct2table(data);
data1.well(:,1)=data1.well(:,1)+960;

%to check numel in categories
numel(data1.treatment(find(data1.treatment==3)));

%create a idx list accordingly
idxlist = [921541:1105488]'; % +1 for starting idx and total - 1 for last
data1 = [data1, array2table(idxlist(:,1))];

%rename the column name
data1.Properties.VariableNames([4]) = {'idxlist'};

%actual_data = [actual_data; dcda0];
%save('labelled_trace_306k_bigdata_with_red.mat', 'data', 'time','-v7.3')

%To find all the CCCP, DMSO & Apramycin Wells
unique(data1.well(find(data1.treatment==0)));

idx_dmso = [ 1,  2,  3,  5,  6,  7,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,...
21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40,...
41, 42, 43, 44, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 61, 62,...
63, 64, 65, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81,...
82, 85, 86, 87, 88, 89, 90, 92, 93, 95];

idx_cccp = [ 4, 12, 24, 46, 56, 60, 72, 83, 91];

idx_apramycin = [ 8, 34, 36, 45, 48, 66, 84, 94, 96];

addition = 959;
idx_dmso = idx_dmso +addition; idx_cccp = idx_cccp + addition; idx_apramycin = idx_apramycin + addition;
% Joel's Plate Preprocessing

data1.treatment(1)=0;

indx = [];
for i = idx_cccp %CHANGE THIS
indx = [indx find(data1.well==i)'];
end
data1.treatment(indx)=1;

testar = [name_apra cellar_apra];
[ranks, idx] = sort(testar(:,1));
apra_like = [ranks(:,1) testar(idx,2)];

[ranks, idx] = sort(testar(:,1));
cccp_like = [ranks(:,1) testar(idx,2) testar(idx,3)];

name_apra = [];
for i = 1:53
name_apra = [name_apra unique(data1.chem(find(strcmp(data1.treatment,cellar_apra(i)))))];
i = i+1;
end
name_apra = name_apra';
%% Randomize data and put to Train, Test and Val for corresponding labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DMSO Traces
[trainInd0,valInd0,testInd0] = dividerand(size(find(data1.treatment==0),1),0.7, 0.15,0.15);
label_0_data = data(find(data1.treatment==0));

%DATA FOR LABEL 0
label_0_data = label_0_data(trainInd0);
label_0_val = label_0_data(valInd0);
label_0_test = label_0_data(testInd0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CCCP Traces
[trainInd1,valInd1,testInd1] = dividerand(size(find(data1.treatment==1),1),0.7, 0.15,0.15);
label_1_data = data(find(data1.treatment==1));

%DATA FOR LABEL 1
label_1_train = label_1_data(trainInd1);
label_1_val = label_1_data(valInd1);
label_1_test = label_1_data(testInd1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Apramycin Traces
[trainInd2,valInd2,testInd2] = dividerand(size(find(data1.treatment==2),1),0.7, 0.15,0.15);
label_2_data = data(find(data1.treatment==2));

%DATA FOR LABEL 2
label_2_train = label_2_data(trainInd2);
label_2_val = label_2_data(valInd2);
label_2_test = label_2_data(testInd2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Corrupted Traces
[trainInd3,valInd3,testInd3] = dividerand(size(find(data1.treatment==3),1),0.7, 0.15,0.15);
label_3_data = data(find(data1.treatment==3));

%DATA FOR LABEL 3
label_3_train = label_3_data(trainInd3);
label_3_val = label_3_data(valInd3);
label_3_test = label_3_data(testInd3);

%% Generate Red and Green Images for Traces for Label 0 - change folder name % label_0_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_0_val,1)
    dataset = [dataset; horzcat(label_0_val(i).intensG', label_0_val(i).intensR')];
    dataset_labels = [dataset_labels label_0_val(i).treatment];
    dataset_idx = [dataset_idx label_0_val(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['0_val_gr/' ])
mkdir(['0_val_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_0_val,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('0_val_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('0_val_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end

fprintf("\nDone 0_val")


% Generate Red and Green Images for Traces for Label 1 - change label_1_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_1_val,1)
    dataset = [dataset; horzcat(label_1_val(i).intensG', label_1_val(i).intensR')];
    dataset_labels = [dataset_labels label_1_val(i).treatment];
    dataset_idx = [dataset_idx label_1_val(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['1_val_gr/' ])
mkdir(['1_val_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_1_val,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('1_val_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('1_val_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 1_val")
% Generate Red and Green Images for Traces for Label 2- change label_2_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_2_val,1)
    dataset = [dataset; horzcat(label_2_val(i).intensG', label_2_val(i).intensR')];
    dataset_labels = [dataset_labels label_2_val(i).treatment];
    dataset_idx = [dataset_idx label_2_val(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['2_val_gr/' ])
mkdir(['2_val_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_2_val,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('2_val_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('2_val_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 2_val")

% Generate Red and Green Images for Traces for Label 3- change label_3_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_3_val,1)
    dataset = [dataset; horzcat(label_3_val(i).intensG', label_3_val(i).intensR')];
    dataset_labels = [dataset_labels label_3_val(i).treatment];
    dataset_idx = [dataset_idx label_3_val(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['3_val_gr/' ])
mkdir(['3_val_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_3_val,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('3_val_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('3_val_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 3_val")

% TEST DATA IMAGES
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_0_test,1)
    dataset = [dataset; horzcat(label_0_test(i).intensG', label_0_test(i).intensR')];
    dataset_labels = [dataset_labels label_0_test(i).treatment];
    dataset_idx = [dataset_idx label_0_test(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['0_test_gr/' ])
mkdir(['0_test_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_0_test,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('0_test_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('0_test_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 0_test")
% Generate Red and Green Images for Traces for Label 1 - change label_1_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_1_test,1)
    dataset = [dataset; horzcat(label_1_test(i).intensG', label_1_test(i).intensR')];
    dataset_labels = [dataset_labels label_1_test(i).treatment];
    dataset_idx = [dataset_idx label_1_test(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['1_test_gr/' ])
mkdir(['1_test_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_1_test,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('1_test_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('1_test_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 1_test")
% Generate Red and Green Images for Traces for Label 2- change label_2_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_2_test,1)
    dataset = [dataset; horzcat(label_2_test(i).intensG', label_2_test(i).intensR')];
    dataset_labels = [dataset_labels label_2_test(i).treatment];
    dataset_idx = [dataset_idx label_2_test(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['2_test_gr/' ])
mkdir(['2_test_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_2_test,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('2_test_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('2_test_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 2_test")

% Generate Red and Green Images for Traces for Label 3- change label_3_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_3_test,1)
    dataset = [dataset; horzcat(label_3_test(i).intensG', label_3_test(i).intensR')];
    dataset_labels = [dataset_labels label_3_test(i).treatment];
    dataset_idx = [dataset_idx label_3_test(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['3_test_gr/' ])
mkdir(['3_test_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_3_test,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('3_test_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('3_test_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 3_test")

% TRAIN DATA IMAGES
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_0_data,1)
    dataset = [dataset; horzcat(label_0_data(i).intensG', label_0_data(i).intensR')];
    dataset_labels = [dataset_labels label_0_data(i).treatment];
    dataset_idx = [dataset_idx label_0_data(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['0_train_gr/' ])
mkdir(['0_train_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_0_data,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('0_train_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('0_train_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 0_train")
% Generate Red and Green Images for Traces for Label 1 - change label_1_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_1_train,1)
    dataset = [dataset; horzcat(label_1_train(i).intensG', label_1_train(i).intensR')];
    dataset_labels = [dataset_labels label_1_train(i).treatment];
    dataset_idx = [dataset_idx label_1_train(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['1_train_gr/' ])
mkdir(['1_train_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_1_train,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('1_train_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('1_train_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 1_train")
% Generate Red and Green Images for Traces for Label 2- change label_2_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_2_train,1)
    dataset = [dataset; horzcat(label_2_train(i).intensG', label_2_train(i).intensR')];
    dataset_labels = [dataset_labels label_2_train(i).treatment];
    dataset_idx = [dataset_idx label_2_train(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['2_train_gr/' ])
mkdir(['2_train_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_2_train,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('2_train_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('2_train_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 2_train")

% Generate Red and Green Images for Traces for Label 3- change label_3_(train,val,test) accordingly
dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_3_train,1)
    dataset = [dataset; horzcat(label_3_train(i).intensG', label_3_train(i).intensR')];
    dataset_labels = [dataset_labels label_3_train(i).treatment];
    dataset_idx = [dataset_idx label_3_train(i).idxlist];
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['3_train_gr/' ])
mkdir(['3_train_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_3_train,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('3_train_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('3_train_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 3_train")

%% Train, Test & Val Paths

%Train
redtr0 = dir('.\0_train_re\*.jpg');
redtr1 = dir('.\1_train_re\*.jpg');
redtr2 = dir('.\2_train_re\*.jpg');
redtr3 = dir('.\3_train_re\*.jpg');

greentr0 = dir('.\0_train_gr\*.jpg');
greentr1 = dir('.\1_train_gr\*.jpg');
greentr2 = dir('.\2_train_gr\*.jpg');
greentr3 = dir('.\3_train_gr\*.jpg');

%Test
redtest0 = dir('.\0_test_re\*.jpg');
redtest1 = dir('.\1_test_re\*.jpg');
redtest2 = dir('.\2_test_re\*.jpg');
redtest3 = dir('.\3_test_re\*.jpg');

greentest0 = dir('.\0_test_gr\*.jpg');
greentest1 = dir('.\1_test_gr\*.jpg');
greentest2 = dir('.\2_test_gr\*.jpg');
greentest3 = dir('.\3_test_gr\*.jpg');

%Val
redval0 = dir('.\0_val_re\*.jpg');
redval1 = dir('.\1_val_re\*.jpg');
redval2 = dir('.\2_val_re\*.jpg');
redval3 = dir('.\3_val_re\*.jpg');

greenval0 = dir('.\0_val_gr\*.jpg');
greenval1 = dir('.\1_val_gr\*.jpg');
greenval2 = dir('.\2_val_gr\*.jpg');
greenval3 = dir('.\3_val_gr\*.jpg');

%% Generate 3 Channel Images for all Labels - change the folder names and red(train,test,val) 

% TRAIN

%Label 0
mkdir(['0_train_final/'])
for ii = 1:length(redtr0)
    currentfile = redtr0(ii).name;
    currentimg = imread(fullfile('.\0_train_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentr0(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\0_train_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('0_train_final', ...
               [sprintf('%s',num2str(redtr0(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 0_train");
%Label 1
mkdir(['1_train_final/'])
for ii = 1:length(redtr1)
    currentfile = redtr1(ii).name;
    currentimg = imread(fullfile('.\1_train_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentr1(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\1_train_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('1_train_final', ...
               [sprintf('%s',num2str(redtr1(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 1_train");
%Label 2
mkdir(['2_train_final/'])
for ii = 1:length(redtr2)
    currentfile = redtr2(ii).name;
    currentimg = imread(fullfile('.\2_train_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentr2(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\2_train_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('2_train_final', ...
               [sprintf('%s',num2str(redtr2(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 2_train");
%Label 3
mkdir(['3_train_final/'])
for ii = 1:length(redtr3)
    currentfile = redtr3(ii).name;
    currentimg = imread(fullfile('.\3_train_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentr3(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\3_train_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('3_train_final', ...
               [sprintf('%s',num2str(redtr3(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 3_train");
%VAL

%Label 0
mkdir(['0_val_final/'])
for ii = 1:length(redval0)
    currentfile = redval0(ii).name;
    currentimg = imread(fullfile('.\0_val_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greenval0(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\0_val_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('0_val_final', ...
               [sprintf('%s',num2str(redval0(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 0_val");
%Label 1
mkdir(['1_val_final/'])
for ii = 1:length(redval1)
    currentfile = redval1(ii).name;
    currentimg = imread(fullfile('.\1_val_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greenval1(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\1_val_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('1_val_final', ...
               [sprintf('%s',num2str(redval1(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 1_val");
%Label 2
mkdir(['2_val_final/'])
for ii = 1:length(redval2)
    currentfile = redval2(ii).name;
    currentimg = imread(fullfile('.\2_val_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greenval2(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\2_val_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('2_val_final', ...
               [sprintf('%s',num2str(redval2(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 2_val");
%Label 3
mkdir(['3_val_final/'])
for ii = 1:length(redval3)
    currentfile = redval3(ii).name;
    currentimg = imread(fullfile('.\3_val_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greenval3(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\3_val_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('3_val_final', ...
               [sprintf('%s',num2str(redval3(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 3_val");
% TEST

%Label 0
mkdir(['0_test_final/'])
for ii = 1:length(redtest0)
    currentfile = redtest0(ii).name;
    currentimg = imread(fullfile('.\0_test_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentest0(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\0_test_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('0_test_final', ...
               [sprintf('%s',num2str(redtest0(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 0_test");
%Label 1
mkdir(['1_test_final/'])
for ii = 1:length(redtest1)
    currentfile = redtest1(ii).name;
    currentimg = imread(fullfile('.\1_test_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentest1(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\1_test_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('1_test_final', ...
               [sprintf('%s',num2str(redtest1(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 1_test");
%Label 2
mkdir(['2_test_final/'])
for ii = 1:length(redtest2)
    currentfile = redtest2(ii).name;
    currentimg = imread(fullfile('.\2_test_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentest2(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\2_test_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('2_test_final', ...
               [sprintf('%s',num2str(redtest2(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 2_test");
%Label 3
mkdir(['3_test_final/'])
for ii = 1:length(redtest3)
    currentfile = redtest3(ii).name;
    currentimg = imread(fullfile('.\3_test_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentest3(ii).name; %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\3_test_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('3_test_final', ...
               [sprintf('%s',num2str(redtest3(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 3_test");

%% FAST ARRAY APPEND

dataset = [];
dataset_labels = [];
dataset_idx = [];
dataset_test = [];
for i = 1:size(label_1_train,1)
    dataset(end+1,:) = horzcat(label_1_train(i).intensG', label_1_train(i).intensR');
    dataset_labels(end+1) = label_1_train(i).treatment;
    dataset_idx(end+1) = label_1_train(i).idxlist;
    if mod(i,100)==0
        i
    end
end
dataset_test = [dataset_labels' dataset_idx' dataset];
mkdir(['1_train_gr/' ])
mkdir(['1_train_re/' ])
feat_gr= [dataset_test(:,3:452)];
feat_re= [dataset_test(:,453:end)];
for sample = 1:size(label_1_train,1)
    gr = RP(feat_gr(sample,:),3,4);
    re = RP(feat_re(sample,:),3,4);
    imwrite(gr, fullfile('1_train_gr', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
    imwrite(re, fullfile('1_train_re', ...
           [sprintf('%02d-%1d-%1d',dataset_idx(sample),data(dataset_idx(sample)).well,dataset_labels(sample)) '.jpg'])); 
end
fprintf("\nDone 1_train")

redtr1 = dir('.\1_train_re\*.jpg');
greentr1 = dir('.\1_train_gr\*.jpg');

mkdir(['1_train_final/'])
for ii = 1:length(redtr1)
    currentfile = redtr1(ii).name;
    currentimg = imread(fullfile('.\1_train_re\',currentfile));
    redtr0img{ii} = currentimg;
    
    currentfile1 = greentr1(ii).name;  %CHANGE GREENTR{1,2,3}
    currentimg1 = imread(fullfile('.\1_train_gr\',currentfile1));
    greentr0img{ii} = currentimg1;
    
    redCH = redtr0img{ii}(:,:,1);
    redtr0img{ii}(:,:,2) = greentr0img{ii}(:,:,1);
    greenCH = redtr0img{ii}(:,:,2);
    redtr0img{ii}(:,:,3) = 0;
    blueCH = redtr0img{ii}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('1_train_final', ...
               [sprintf('%s',num2str(redtr1(ii).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
end
fprintf("\n\nDONE 1_train");

%% FOR Categorizing to WELLS - test  data
dataset = [];
dataset_well = [];
dataset_treat = [];
dataset_idx = [];
dataset_test = [];
% For label 0 Test
for i = 1:size(label_0_test,1)
    dataset(end+1,:) = horzcat(label_0_test(i).intensG', label_0_test(i).intensR');
    dataset_well(end+1) =  label_0_test(i).well;
    dataset_treat(end+1) =  label_0_test(i).treatment;
    dataset_idx(end+1) = label_0_test(i).idxlist;
    if mod(i,100)==0
        i
    end
end
% For label 1 Test
%dataset_test = [dataset_well' dataset_treat' dataset_idx' dataset];
for i = 1:size(label_1_test,1)
    dataset(end+1,:) = horzcat(label_1_test(i).intensG', label_1_test(i).intensR');
    dataset_well(end+1) =  label_1_test(i).well;
    dataset_treat(end+1) =  label_1_test(i).treatment;
    dataset_idx(end+1) = label_1_test(i).idxlist;
    if mod(i,100)==0
        i
    end
end
%dataset_test = [dataset_well' dataset_treat' dataset_idx' dataset];
% For label 2 Test
for i = 1:size(label_2_test,1)
    dataset(end+1,:) = horzcat(label_2_test(i).intensG', label_2_test(i).intensR');
    dataset_well(end+1) =  label_2_test(i).well;
    dataset_treat(end+1) =  label_2_test(i).treatment;
    dataset_idx(end+1) = label_2_test(i).idxlist;
    if mod(i,100)==0
        i
    end
end
%dataset_test = [dataset_well' dataset_treat' dataset_idx' dataset];
%For label 3 Test
for i = 1:size(label_3_test,1)
    dataset(end+1,:) = horzcat(label_3_test(i).intensG', label_3_test(i).intensR');
    dataset_well(end+1) =  label_3_test(i).well;
    dataset_treat(end+1) =  label_3_test(i).treatment;
    dataset_idx(end+1) = label_3_test(i).idxlist;
    if mod(i,100)==0
        i
    end
end
dataset_test = [dataset_well' dataset_treat' dataset_idx' dataset];

%generate folders for well 
la = dataset_test(:,1);
feat_gr= [dataset_test(:,4:453)];
feat_re= [dataset_test(:,454:end)];
n_class = max(la); 

for c = 1: n_class
    c
    [indx, c_la] = find(la==c);
    mkdir(['well_test_gr/' num2str(c)]);  % make a folder
    mkdir(['well_test_re/' num2str(c)]);
    for sample = 1:size(indx,1)
        gr = RP(feat_gr(indx(sample),:), 3,4);
        re = RP(feat_re(indx(sample),:),3,4);
        imwrite(gr, fullfile('well_test_gr', num2str(c), ...
               [sprintf('%02d-%03d-%1d',dataset_test(indx(sample),3),dataset_test(indx(sample),1),dataset_test(indx(sample),2)) '.jpg'])); 
        imwrite(re, fullfile('well_test_re', num2str(c), ...
                [sprintf('%02d-%03d-%1d',dataset_test(indx(sample),3),dataset_test(indx(sample),1),dataset_test(indx(sample),2)) '.jpg'])); 
    end
end

%combine images for each well folder


%For Both the below loops cd to the directory and run snippets & remove
%first 2 rows
dirinfo_gr = dir('C:\Users\Kraljlab\Desktop\BigData_CNN_ImagesModel\well_test_gr');
subdirinfo_gr = cell(length(dirinfo_gr),1);
for K = 1 : length(dirinfo_gr)
  thisdir = dirinfo_gr(K).name;
  subdirinfo_gr{K} = dir(fullfile('C:\Users\Kraljlab\Desktop\BigData_CNN_ImagesModel\well_test_gr',thisdir, '*.jpg'));
end

dirinfo_re = dir('C:\Users\Kraljlab\Desktop\BigData_CNN_ImagesModel\well_test_re');
subdirinfo_re = cell(length(dirinfo_re),1);
for K = 1 : length(dirinfo_re)
  thisdir = dirinfo_re(K).name;
  subdirinfo_re{K} = dir(fullfile('C:\Users\Kraljlab\Desktop\BigData_CNN_ImagesModel\well_test_re',thisdir, '*.jpg'));
end

%make folders first 
for c  = 10:96
    mkdir(['well_test_final/' num2str(c)]);
end

%make folder according to number of wells
for c  = 1:480
    mkdir(['well_test_final/' num2str(sprintf('%03d',c))]);
end

%run for placing traces to its respective wells
for c  = 1:480
    c
    %read green imgs
    for i = 1:numel(subdirinfo_gr{c,1})
    currentfile = subdirinfo_gr{c,1}(i).name;
    currentimg = imread(char(fullfile(strcat(subdirinfo_gr{c,1}(i).folder ,{'\'}, currentfile ))));
    greentr0img{i} = currentimg;
    %read red imgs
    currentfile1 = subdirinfo_re{c,1}(i).name;
    currentimg1 = imread(char(fullfile(strcat(subdirinfo_re{c,1}(i).folder ,{'\'}, currentfile1 ))));
    redtr0img{i} = currentimg1;
    %combine now
    redCH = redtr0img{i}(:,:,1);
    redtr0img{i}(:,:,2) = greentr0img{i}(:,:,1);
    greenCH = redtr0img{i}(:,:,2);
    redtr0img{i}(:,:,3) = 0;
    blueCH = redtr0img{i}(:,:,3);
    imwrite(cat(3,redCH,greenCH,blueCH), fullfile('well_test_final', subdirinfo_gr{c,1}(i).name(end-8:end-6), ...
               [sprintf('%s',num2str(subdirinfo_gr{c,1}(i).name(1:end-4))) '.jpg']));
    clear redCH greenCH blueCH currentimg currentimg1 redtr0img greentr0img
    end
end

%% To Generate Excel Sheet for Train, Val, Test Data

%FOR LSTM CNN- Add well field too.
for i=1:154646
data(i).intensG = rem_pbleach(data(i).intensG,50) - mean(rem_pbleach(data(i).intensG,50));
%data(i).intensR = rem_pbleach(data(i).intensR,50) - mean(rem_pbleach(data(i).intensR,50));
end
dataset1 = [];
dataset_labels = [];
dataset_test1 = [];
for i = 1:size(data,1)
    dataset1(end+1,:) = data(i).intensG';
    dataset_labels(end+1) = data(i).well;
    if mod(i,100)==0
        i
    end
end
dataset_test1 = [dataset_labels' dataset1];
   
writetable(array2table(dataset_test1),'conc_data_2_154k_TEST_WELL.csv','Delimiter',',')

%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DMSO Traces
trainInd0 = randsample(size(find(data1.treatment==0),1),57749);
label_0_data = data(find(data1.treatment==0));

%DATA FOR LABEL 0
label_0_data = label_0_data(trainInd0);

%For Random Traces over all traces
[trainInd0,valInd0,testInd0] = dividerand(57749,0.7, 0.15,0.15);
label_0_val = label_0_data(valInd0);
label_0_test = label_0_data(testInd0);
label_0_train = label_0_data(trainInd0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CCCP Traces
trainInd1 = randsample(size(find(data1.treatment==1),1),57749);
label_1_data = data(find(data1.treatment==1));

label_1_data = label_1_data(trainInd1);
%DATA FOR LABEL 1
[trainInd1,valInd1,testInd1] = dividerand(57749,0.7, 0.15,0.15);
label_1_val = label_1_data(valInd1);
label_1_test = label_1_data(testInd1);
label_1_train = label_1_data(trainInd1);

% label_1_val = label_1_train(valInd1);
% label_1_test = label_1_train(testInd1);
% label_1_train = label_1_train(trainInd1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Apramycin Traces
trainInd2 = randsample(size(find(data1.treatment==2),1),57749);
label_2_data = data(find(data1.treatment==2));

label_2_data = label_2_data(trainInd2);
%DATA FOR LABEL 2
[trainInd2,valInd2,testInd2] = dividerand(57749,0.7, 0.15,0.15);
label_2_train = label_2_data(trainInd2);
label_2_val = label_2_data(valInd2);
label_2_test = label_2_data(testInd2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Corrupted Traces
[trainInd3,valInd3,testInd3] = dividerand(size(find(data1.treatment==3),1),0.7, 0.15,0.15);
label_3_data = data(find(data1.treatment==3));

%DATA FOR LABEL 3
label_3_train = label_3_data(trainInd3);
label_3_val = label_3_data(valInd3);
label_3_test = label_3_data(testInd3);

%% Find Common Hits between GC & Model

dr_list = dir('.\Hits_GC\*.xlsx');


[~,txt,~] = xlsread('.\CCCP_APRA_list_confidence.xlsx');
model_name = {};
for j = 4:size(txt,1)
    model_name = [model_name txt{j,2}(3:end)];
    
end
model_name = model_name';

find(intersect(model_name,names))

common_names = {};
for j = 1:size(dr_list,1)
    [~,txt1,~] = xlsread(fullfile('./Hits_GC/',dr_list(j).name));
    cellar1 = {};
    for i = 2:size(txt1,1)
        cellar1 = [cellar1 txt1{i,1}];
    end
    j
    cellar1 = cellar1';
    common_names = [common_names; dr_list(j).name];
    common_names = [common_names; intersect(cellar1,trunc_cellar_ap)];
end
names = unique(common_names);

% For retrieving names
name = [];
for i = 1:size(cellar,1)
name = [name unique(data1.chem(find(strcmp(data1.treatment,cellar(i)))))];
i = i+1;
end
name = name';
% name_ap = sort(name_ap);
testar = [name cellar];
[ranks, idx] = sort(testar(:,1));
ranks = [ranks(:,1) testar(idx,2)];

trunc_cellar={};
for i = 1:size(cellar,1)
trunc_cellar = [trunc_cellar; cellar{i}(3:end)];
end

%% Run for Each drug and save the testar 

dr_list = dir('.\Hits_GC\*.xlsx');

cell_cp = {'CCCPA05';'CCCPA11';'CCCPB01';'CCCPC07';'CCCPC12';'CCCPD11';'CCCPD05';'CCCPD01';'CCCPE02';'CCCPE04';'CCCPE06';'CCCPE08';'CCCPE12';'CCCPF09';'CCCPF07';'CCCPF06';'CCCPF01';'CCCPG07';'CCCPG08';'CCCPH10';'CCCPH02';'CCCPH01';'CCCPF01';'CCCPG12';'PC01H01';'PC01G12';'PC01F01';'PC01D01';'PC01C06';'PC01C12';'PC01B01';'PC01A12';'PC01F01';'PC02H01';'PC02F01';'PC02E12';'PC02C12';'PC02B01';'PC02A12';'PC03H01';'PC03G12';'PC03F01';'PC03E12';'PC03D01';'PC03C12';'PC03A12';'PC03F08';'PC04G12';'PC04C12';'PC04B01';'PC04A12';'PC04H01';'PC05C12';'PC05F01';'PC06A12';'PC06H01';'PC07E12';'PC07D01';'PC07C12';'PC07B01';'PC07A07';'PC07A12';'PC07H01';'PC08G12';'PC08E12';'PC08D01';'PC08C12';'PC08B01';'PC08A12';'PC08F01';'PC08G12';'PC08H01';'PC09G12';'PC09D01';'PC09C12';'PC09B01';'PC09A12';'PC09E07';'PC10D01';'PC10C12';'PC10B01';'PC10A12';'PC10C12';'PC10D01';'PC10E12';'PC10F08';'PC10F01';'PC10G12';'PC10H01';'PC11G12';'PC11F01';'PC11E12';'PC11D01';'PC11C12';'PC11B01';'PC11A12';'PC11D01';'PC11E12';'PC11G12';'PC11H01';'PC12E12';'PC12D01';'PC12C12';'PC12B01';'PC12A12';'PC12C12';'PC12D01';'PC12E12';'PC12F01';'PC12G12';'PC12H01';'PC13H01';'PC13G12';'PC13F01';'PC13E12';'PC13D01';'PC13B01';'PC13A12';'PC13E12';'PC13F01';'PC13G12';'PC13H01';'PC14H01';'PC14G12';'PC14F01';'PC14E12';'PC14D01';'PC14C12';'PC14B01';'PC14A12';'PC14G12';'PC14H01';'PC15E12';'PC15D01';'PC15C12';'PC15B01';'PC15A12';'PC15F01';'PC15G12';'PC15H01'};
cell_apr = {'PC01G03';'PC01G09';'PC01F11';'PC01C08';'PC01A01';'PC01A07';'PC01A10';'PC01E11';'PC02F09';'PC02D01';'PC02C08';'PC02B04';'PC02G08';'PC02H10';'PC03B11';'PC03B10';'PC03B09';'PC04A02';'PC05H10';'PC05H09';'PC05H06';'PC05G04';'PC05G05';'PC05G06';'PC05G09';'PC05F08';'PC05A02';'PC05G09';'PC06A05';'PC06A10';'PC06B04';'PC06C03';'PC06F09';'PC07C07';'PC07C07';'PC07G05';'PC08F02';'PC08F02';'PC09D07';'PC09E01';'PC10G08';'PC10D09';'PC10D09';'PC10G08';'PC12G09';'PC12A11';'PC12G01';'PC13H06';'PC13E06';'PC13C09';'PC13E06';'PC13H12';'PC14G06'};

%FOR APR

trunc_cellar_ap={};
for i = 1:size(cell_apr,1)
trunc_cellar_ap = [trunc_cellar_ap; cell_apr{i}(3:end)];
end

% common_names = {};
% for j = 1:size(dr_list,1)
%     [~,txt1,~] = xlsread(fullfile('./Hits_GC/',dr_list(j).name));
%     cellar1 = {};
%     for i = 2:size(txt1,1)
%         cellar1 = [cellar1 txt1{i,1}];
%     end
%     j
%     cellar1 = cellar1';
%     common_names = [common_names; dr_list(j).name];
%     common_names = [common_names; intersect(cellar1,trunc_cellar_ap)];
% end

names_apra = [];
for i = 1:size(cell_apr,1)
names_apra = [names_apra unique(data1.chem(find(strcmp(data1.treatment,cell_apr(i)))))];
end
names_apra = names_apra';

testar = [names_apra cell_apr];
[ranks, idx] = sort(testar(:,1));
testar = [ranks(:,1) testar(idx,2)];

co = 3;
for j = 1:size(dr_list,1)
    [~,txt1,~] = xlsread(fullfile('./Hits_GC/',dr_list(j).name));
    for i =1:size(testar,1)
        for k = 2:size(txt1,1)
            if (char(txt1(k,1)) == testar{i,2}(3:end))
                testar{i,co} =  dr_list(j).name;
            end
        end
    end
    co = co+1;
end

%FOR CCCP
trunc_cellar={};
for i = 1:size(cell_cp,1)
trunc_cellar = [trunc_cellar; cell_cp{i}(3:end)];
end

names_cp = [];
for i = 1:size(cell_cp,1)
names_cp = [names_cp unique(data1.chem(find(strcmp(data1.treatment,cell_cp(i)))))];
end
names_cp = names_cp';

testar = [names_cp cell_cp];
[ranks, idx] = sort(testar(:,1));
testar = [ranks(:,1) testar(idx,2)];

co = 3;
for j = 1:size(dr_list,1)
    [~,txt1,~] = xlsread(fullfile('./Hits_GC/',dr_list(j).name));
    for i =1:size(testar,1)
        for k = 2:size(txt1,1)
            if (char(txt1(k,1)) == testar{i,2}(3:end))
                testar{i,co} =  dr_list(j).name;
            end
        end
    end
    co = co+1;
end