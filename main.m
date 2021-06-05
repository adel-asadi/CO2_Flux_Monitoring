%% Loading and Cleaning Datasets:

load('EL_stack_res2.mat');
Bnew_fs=Bnew;
b=Rnew;
hs=[38 40 42 44 46 48];
b(:,hs)=[];
Bnew_removed=Bnew_fs(hs);
Bnew_fs(hs)=[];
for i=1:length(b(1,:))
    TF = isnan(b(:,i));
    k=find(TF==1);
    h(i)=length(k);
end
hh=find(h>500);
a=hh;
b(:,a)=[];
Bnew_removed={Bnew_removed Bnew_fs(a)};
Bnew_fs(a)=[];
hss=[8 14 20 24];
b(:,hss)=[];
Bnew_removed={Bnew_removed Bnew_fs(hss)};
Bnew_fs(hss)=[];
d=[];
for i=1:length(b(1,:))
    ccc = isnan(b(:,i));
    d=[d;find(ccc==1)];
end
e=unique(d);
b(e,:)=[];

%% Classification Data Preparation:

bb=b;
g1=find(b(:,1)==0);
bb(g1,1)=0;
g2=find(b(:,1)>0);
bb(g2,1)=1;
bb_norm=normalize(bb(:,2:end));
for i=1:(length(b(1,:))-1)
    v=bb(:,i+1);
    m_c(i)=mean(v);
    std_c(i)=std(v);
end
bbb=[b(:,1) bb_norm];
bb_norm=[bb(:,1) bb_norm];
r=rand(1,400);
[aa,c]=sort(r);
cc=round(c.*28.8);
test_class=bb_norm(cc,:);
true_class_reg=b(cc,1);
bbb_reg=bbb(cc,:);
bb_norm(cc,:)=[];
data_class=bb_norm;
t=find(true_class_reg>0);
test_reg=bbb_reg(t,:);
true_reg=true_class_reg(t,1);

%% Regression Data Preparation:

bbb_norm=b;
g22=find(bbb_norm(:,1)>0);
bbb_norm=bbb_norm(g22,:);
for i=1:(length(b(1,:))-1)
    vv=bbb_norm(:,i+1);
    m_r(i)=mean(vv);
    std_r(i)=std(vv);
end
sss=bbb_norm(:,1);
bbb_norm=bbb;
bbb_norm(cc,:)=[];
g22=find(bbb_norm(:,1)>0);
bbb_norm=bbb_norm(g22,:);
ss=bbb_norm(:,1);
bbb_normm=bbb_norm;
for i=1:(length(b(1,:))-1)
    kk=bbb_normm(:,i+1);
    kkk=(kk.*std_c(i))+m_c(i);
    kkkk=(kkk-m_r(i))./std_r(i);
    bbb_normm(:,i+1)=kkkk;
end
data_reg=bbb_normm;
data_reg(:,1)=((data_reg(:,1))-mean(sss))./std(sss);
for i=1:(length(b(1,:))-1)
    tt=test_reg(:,i+1);
    ttt=(tt.*std_c(i))+m_c(i);
    tttt=(ttt-m_r(i))./std_r(i);
    test_reg(:,i+1)=tttt;
end
test_reg(:,1)=((test_reg(:,1))-mean(sss))./std(sss);

%% Feature Selection for Classification:

mdl = fscnca(data_class(:,2:end),data_class(:,1),'Standardize',true);
figure()
plot(mdl.FeatureWeights,'ro')
grid on
hold on
plot(0:length(b(1,:))+1,ones(length(b(1,:))+2)*0.1,'r','LineWidth',1)
xlabel('Feature index')
ylabel('Feature weight')
title('Feature Selection for Classification by NCA Method')
hold off
class_w=mdl.FeatureWeights;
wc=[1;(find(class_w>0.1)+1)];
Bnew_fs(1)=[];
Bnew_fswc=Bnew_fs(wc(2:end)-1);
wcf=class_w(wc(2:end)-1);
wcf(:,2)=1:length(wcf);
wcf_desc=sortrows(wcf,'descend');
Bnew_fswc_desc=Bnew_fswc(wcf_desc(:,2));
train_class=data_class(:,wc);

%% Feature Selection for Regression:

mdl = fsrnca(data_reg(:,2:end),data_reg(:,1),'Standardize',true);
figure()
plot(mdl.FeatureWeights,'ro')
grid on
hold on
plot(0:length(b(1,:))+1,ones(length(b(1,:))+2)*0.1,'r','LineWidth',1)
xlabel('Feature index')
ylabel('Feature weight')
title('Feature Selection for Regression by NCA Method')
hold off
reg_w=mdl.FeatureWeights;
wr=[1;(find(reg_w>0.1)+1)];
Bnew_fswr=Bnew_fs(wr(2:end)-1);
wrf=reg_w(wr(2:end)-1);
wrf(:,2)=1:length(wrf);
wrf_desc=sortrows(wrf,'descend');
Bnew_fswr_desc=Bnew_fswr(wrf_desc(:,2));
train_reg=data_reg(:,wr);

%% Classification & Regression Models Training:

load('Models.mat');
%[trainedModel_c, validationAccuracy] = trainClassifier(train_class);
%[trainedModel_r, validationRMSE] = trainRegressionModel(train_reg);
%trainedModel_c = fitcensemble(train_class(:,2:end),train_class(:,1),'NumLearningCycles',1000,'Method','GentleBoost');
%trainedModel_r = fitrensemble(train_reg(:,2:end),train_reg(:,1),'NumLearningCycles',1000,'Method','LSBoost');

%% Classification Model Testing:

yfit_c = trainedModel_c.predictFcn(test_class(:,wc(2:end)));
%yfit_c = predict(trainedModel_c,test_class(:,wc(2:end)));
targets=test_class(:,1);
outputs=yfit_c;
%figure;plotconfusion(targets,outputs,'Classification Testing Accuracy')
figure;confusionchart(targets,outputs);
title('Classification Testing Accuracy')
err_c=length(find((targets-outputs)==0))/(length(test_class(:,1)));

%% Regression Model Testing:

yfit_r = trainedModel_r.predictFcn(test_reg(:,wr(2:end)));
%yfit_r = predict(trainedModel_r,test_reg(:,wr(2:end)));
targs1=test_reg(:,1);
outs1=yfit_r;
figure;
plotregression(targs1,outs1,'Regression Testing Accuracy')
xx=(yfit_r*std(sss)+mean(sss));
figure;
plotregression(true_reg,xx,'Regression Testing Accuracy')
xlabel('Observed CO_2 Flux (g m^-^2 day^-^1)')
ylabel('Model Predicted CO_2 Flux (g m^-^2 day^-^1)')
r_sq=min(corrcoef(targs1,outs1)).^2;
r_sq_t=min(corrcoef(xx,true_reg)).^2;

%% Final Combined Model Testing:

test_reg_f=test_reg;
for i=1:(length(b(1,:))-1)
    xxx=test_reg(:,i+1);
    xxxx=(xxx.*std_r(i))+m_r(i);
    xxxxx=(xxxx-m_c(i))./std_c(i);
    test_reg_f(:,i+1)=xxxxx;
end
test_class_fc=test_class;
test_class_f=test_class;
test_class_fc(t,:)=[];
test_class_f(t,:)=[];
final_check_c=[test_class_fc;test_reg_f];
tr=find(true_class_reg==0);
final_target=[true_class_reg(tr);true_class_reg(t)];
yfit_c_f = trainedModel_c.predictFcn(final_check_c(:,wc(2:end)));
%yfit_c_f = predict(trainedModel_c,final_check_c(:,wc(2:end)));
for i=1:(length(b(1,:))-1)
    yyy=test_class_f(:,i+1);
    yyyy=(yyy.*std_c(i))+m_c(i);
    yyyyy=(yyyy-m_r(i))./std_r(i);
    test_class_f(:,i+1)=yyyyy;
end
final_check_r=[test_class_f;test_reg];
yfit_r_f=yfit_c_f;
zz=find(yfit_c_f(:,1)==1);
yfit_r_f(zz) = trainedModel_r.predictFcn(final_check_r(zz,wr(2:end)));
%yfit_r_f(zz) = predict(trainedModel_r,final_check_r(zz,wr(2:end)));
ww=(yfit_r_f(zz).*std(sss))+mean(sss);
yfit_r_f(zz)=ww;
r_sq_f=min(corrcoef(yfit_r_f,final_target)).^2;
err_rmse = sqrt(immse(yfit_r_f,final_target));
figure;
plotregression(final_target,yfit_r_f,'Final Model Accuracy Check')
xlabel('Observed CO_2 Flux (g m^-^2 day^-^1)')
ylabel('Model Predicted CO_2 Flux (g m^-^2 day^-^1)')

%% The End...
