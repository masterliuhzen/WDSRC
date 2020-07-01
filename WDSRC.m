% =========================================================================
% Zhen liu, Xiao-Jun Wu, Zhenqiu Shu, Hefeng Yin, Zhe Chen
% "Weighted discriminative sparse representation for image classification" in "Neural Processing Letters".
% Written by Zhen Liu @ JNU
% Contact: master_liu@163.com
% The databases in this paper can be available by contacting the first author(master_liu@163.com).
% June, 2020.
% =========================================================================      

clear all;
clc
warning off;

load 'LFW'

fea = double(fea);
nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end

for gama  = [0.00001]  %[100 10 1 0.1 0.01 0.001 0.0001 0.00001]
    for lambda  = [0.01]  %[100 10 1 0.1 0.01 0.001 0.0001 0.00001]
        for sele_num  = 8 %3:8
            for iter = 1:10
                Train_Ma  = [];
                Train_Lab = [];
                Test_Ma   = [];
                Test_Lab  = [];
                for j = 1:nnClass
                    idx      = find(gnd==j);
                    randIdx  = randperm(num_Class(j));
                    Train_Ma = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];
                    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
                    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];
                    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];                 
                end
                Train_Ma = Train_Ma';                       % transform to a sample per column
                Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
                Train_Lab=Train_Lab';
                Test_Ma  = Test_Ma';
                Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]);  %
                Test_Lab = Test_Lab';
                
                [size1 size2]=size(Train_Ma);
                M=eye(sele_num*nnClass);
                for i=1:nnClass
                    xi=Train_Ma(:,(i-1)*sele_num+1:i*sele_num);
                    M((i-1)*sele_num+1:i*sele_num,(i-1)*sele_num+1:i*sele_num)=xi'*xi;
                end
                
                ID = [];
                for indTest = 1:size(Test_Ma,2)
                    G = [];
                    for indTrain= 1:size(Train_Ma,2)
                        %[g]     =   norm(Test_Ma(:,indTest)-Train_Ma(:,indTrain));
                        [g]     =   exp(norm(Test_Ma(:,indTest)-Train_Ma(:,indTrain)));
                        G       =   [G g];
                    end
                    W = G;
                   W = diag(G/max(G));
                    T=inv((1+2*gama)*Train_Ma'*Train_Ma+2*gama*(nnClass-2)*M+lambda*W'*W)*Train_Ma';
                    [id]    =   Classification_WDSRC(Train_Ma,T,Test_Ma(:,indTest),Train_Lab);
                    ID      =   [ID id];
                end
                cornum      =   sum(ID==Test_Lab);
                accracy =   [cornum/length(Test_Lab)];
                acc(iter,1) = accracy;
                fprintf([' gama=' num2str(gama) ' lambda=' num2str(lambda) 'sele_num=' num2str(sele_num) ' Rec=' num2str(accracy)  '\n']);
                
                clearvars j idx Train_Ma Test_Ma Train_Lab Test_Lab size1 size2 M i xi indTest g G W T  id ID cornum Rec
                iter = iter + 1 ;
            end
            % % % % % % % % % % % % % % % % % % %
            Ravg = mean(acc);
            Rstd = std(acc);
            fprintf([ ' gama=' num2str(gama)   ' lambda=' num2str(lambda) 'sele_num=' num2str(sele_num) ' Ravg=' num2str(Ravg)  ' Rstd=' num2str(Rstd)  '\n']);
            % % % % % % % % % % % % % % % %
        end
    end
end
