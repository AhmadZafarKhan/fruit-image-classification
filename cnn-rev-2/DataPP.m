clear;
clc;

% Datapreprocessing. Importing the number of subdirectories in the Training folder
% into an array. 
Trainfiles = dir('D:\Coursework\8\EEE485\Project\fruits\fruits-360\Training'); %Change the dir location to the one in your computer
dirFlags = [Trainfiles.isdir]; % Find dirs
subFolders = Trainfiles(dirFlags); %Save them to struct
subFolders([1 2]) = []; % Remove redundant dirs

%
% In each subdir or class, go through each image and save it in a struct
% named files
%
for i=1:length(subFolders)
    s1(i)= convertCharsToStrings(subFolders(i).name);
    s(i)=strjoin(['D:\Coursework\8\EEE485\Project\fruits\fruits-360\Training\' s1(i) '\'],'');
    files{i} = dir(fullfile(s(i), '*.jpg'));
end

cnt=1; % initialize a count to verify all the data instances are saved

%
% Perform HOG analysis using Piotrs Toolbox Image
% https://github.com/pdollar/toolbox
%
for i=1:length(files)
    for j=1:length(files{i})
        IMG=imread(strjoin([s(i) files{i}(j).name],''));
        I=imResample(single(IMG)/255,[100 100]);
        HOGValuesTRAIN(:,:,:,cnt)=hog(I,5,1);
        HOGValuesTRAINLabel(i,j)=i; % Create a label vector
        cnt=cnt+1
    end
end

%
% Do the same for the Test samples
%
Trainfiles = dir('D:\Coursework\8\EEE485\Project\fruits\fruits-360\Test');
dirFlags = [Trainfiles.isdir];
subFolders = Trainfiles(dirFlags);
subFolders([1 2]) = [];

for i=1:length(subFolders)
    s1(i)= convertCharsToStrings(subFolders(i).name);
    s(i)=strjoin(['D:\Coursework\8\EEE485\Project\fruits\fruits-360\Test\' s1(i) '\'],'');
    files{i} = dir(fullfile(s(i), '*.jpg'));
end

cnt=1;

for i=1:length(files)
    for j=1:length(files{i})
        IMG=imread(strjoin([s(i) files{i}(j).name],''));
        I=imResample(single(IMG)/255,[100 100]);
        HOGValuesTEST(:,:,:,cnt)=hog(I,5,1);
        HOGValuesTESTLabel(i,j)=i;
        cnt=cnt+1
    end
end

% Reformat the label matrix into a label vector corresponding to
% Dataset size
yTrain=0;
for i=1:95    
    m=HOGValuesTRAINLabel(i,:);
    m(m==0)=[];
    yTrain=[yTrain;m'];
end

yTrain(1)=[];

yTest=0;
for i=1:95    
    m=HOGValuesTESTLabel(i,:);
    m(m==0)=[];
    yTest=[yTest;m'];
end

yTest(1)=[];

% save in a .mat file
save('Dataset.mat','HOGValuesTRAIN','HOGValuesTEST','yTrain','yTest');
