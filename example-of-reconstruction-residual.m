 load corn
 XTR=[mp5spec.data];
 XTE=[m5spec.data;mp6spec.data];
 
 [XSelected,XRest,vSelectedRowIndex,vNotSelectedSample]=ks(XTR,round(size(XTR,1)*2/3)); 
 xx=XTR(vSelectedRowIndex,:);
 x3=[XTR(vNotSelectedSample,:);XTE];
 ss=size(XTR(vNotSelectedSample,:),1);
 trueLabels=[zeros(length(vNotSelectedSample),1);ones(size(XTE,1),1)];
 [XSelected,XRest,vSelectedRowIndex,vNotSelectedSample]=ks(xx,round(size(xx,1)/2)); 
 x1=xx(vSelectedRowIndex,:);
 x2=[xx(vNotSelectedSample,:)];

 xtrain=x1;
[coeff1,score1,latent,tsquared,explained,mu1] = pca(xtrain);
cumulative_explained_variance = cumsum(explained);
fc=99;
num_components_needed = find(cumulative_explained_variance >= fc, 1);
ss2=num_components_needed;
t = score1(:,1:ss2)*coeff1(:,1:ss2)' + repmat(mu1,size(xtrain,1),1);
xg2=((x2-repmat(mu1,size(x2,1),1))*coeff1(:,1:ss2)*coeff1(:,1:ss2)'+repmat(mu1,size(x2,1),1));
xc2=x2-xg2;
xg3=(x3-repmat(mu1,size(x3,1),1))*coeff1(:,1:ss2)*coeff1(:,1:ss2)'+repmat(mu1,size(x3,1),1);
xc3=x3-xg3;

commands={@iforest,@ocsvm,@lof}
outs0 = cell(1, length(commands));
for i = 1:length(commands)
   [Mdl1{i},tf1{i},s1{i}]= commands{i}([x1;x2],"ContaminationFraction",0.1);  % 调用第i个命令并获取结果
   [tf_test1{i},s_test1{i}] = isanomaly(Mdl1{i},x3);
   [sn1,sp1,acc1] = snspacc(trueLabels, double(tf_test1{i}));
   outs0{i}=[sn1 sp1 acc1];
end

outs1 = cell(1, length(commands));
for i = 1:length(commands)
   [Mdl2{i},tf2{i},s2{i}]= commands{i}(xc2,"ContaminationFraction",0.1);  % 调用第i个命令并获取结果
   [tf_test2{i},s_test2{i}] = isanomaly(Mdl2{i},xc3);
   [sn2,sp2,acc2] = snspacc(trueLabels, double(tf_test2{i}));
   outs1{i}=[sn2 sp2 acc2];
end
ecochs=20000;
[net] = aec2(xtrain,ecochs)
sxg2=predict(net, x2');
sxc2 = x2-sxg2';
sxg3=predict(net, x3');
sxc3 = x3-sxg3';

outs3 = cell(1, length(commands));
for i = 1:length(commands)
   [Mdl3{i},tf3{i},s3{i}]= commands{i}(sxc2,"ContaminationFraction",0.1);  % 调用第i个命令并获取结果
   [tf_test3{i},s_test3{i}] = isanomaly(Mdl3{i},sxc3);
   [sn3,sp3,acc3] = snspacc(trueLabels, double(tf_test3{i}));
   outs3{i}=[sn3 sp3 acc3];
end


result=[]
for j=1:1:3
shuchu=[outs0{j} outs1{j} outs3{j}];
result=[result;shuchu];   
end
result


