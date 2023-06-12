function [modes,its]=iceemdan(x,Nstd,NR,MaxIter,SNRFlag)

%modes=iceemdan(x,Nstd,NR,MaxIter,SNRFlag)
%[modes its]=iceemdan(x,Nstd,NR,MaxIter,SNRFlag)
%函数的输入参数如下：
%x：输入信号。
%Nstd：停止标准，用于控制最终的模态函数数量。
%NR：模态函数的重构次数，用于计算信噪比。
%MaxIter：最大迭代次数，用于EMD过程。
%SNRFlag：信噪比计算标志，确定是否对噪声进行调整。
%函数的输出结果如下：
%modes：分解得到的本征模态函数。
%its：每个模态函数的迭代次数。

x=x(:)';%转置为行向量
desvio_x=std(x);%计算标准差
x=x/desvio_x;%标准化

aux=zeros(size(x));
iter=zeros(NR,round(log2(length(x))+5));%创建一个大小为 NR 行、round(log2(length(x))+5) 列的全零矩阵 iter，用于存储迭代次数。

white_noise = cell(1,NR);%创建一个大小为 NR 的单元格数组 white_noise，用于存储白噪声实现。
for i=1:NR
    white_noise{i}=randn(size(x));%creates the noise realizations
end

modes_white_noise = cell(1,NR); %创建一个大小为 NR 的单元格数组 modes_white_noise，用于存储白噪声实现的模态函数。
for i=1:NR
    modes_white_noise{i}=emd(white_noise{i});%calculates the modes of white gaussian noise
end

for i=1:NR %calculates the first mode
    xi=x+Nstd*modes_white_noise{i}(1,:)/std(modes_white_noise{i}(1,:));
    [temp, ~, it]=emd(xi,'MAXMODES',1,'MAXITERATIONS',MaxIter);
    temp=temp(1,:);
    aux=aux+(xi-temp)/NR;
    iter(i,1)=it;
end

modes= x-aux; %saves the first mode
medias = aux;
k=1;
aux=zeros(size(x));
es_imf = min(size(emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter)));

while es_imf>1 %calculates the rest of the modes
    for i=1:NR
        tamanio=size(modes_white_noise{i});
        if tamanio(1)>=k+1
            noise=modes_white_noise{i}(k+1,:);
            if SNRFlag == 2
                noise=noise/std(noise); %adjust the std of the noise
            end
            noise=Nstd*noise;
            try
                [temp,~,it]=emd(medias(end,:)+std(medias(end,:))*noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            catch    
                it=0;
                %disp('catch 1 '); disp(num2str(k))
                temp=emd(medias(end,:)+std(medias(end,:))*noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            end
            temp=temp(end,:);
        else
            try
                [temp, ~, it]=emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter);
            catch
                temp=emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter);
                it=0;
                %disp('catch 2 sin ruido')
            end
            temp=temp(end,:);
        end
        aux=aux+temp/NR;
    iter(i,k+1)=it;    
    end
    modes=[modes;medias(end,:)-aux];
    medias = [medias;aux];
    aux=zeros(size(x));
    k=k+1;
    es_imf = min(size(emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter)));
end
modes = [modes;medias(end,:)];
modes=modes*desvio_x;
its=iter;
