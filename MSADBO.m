%% �����㷨��DBO��
function [fMin , bestX, Convergence_curve ] = MSADBO(N, Max_iter,lb,ub,dim,fobj  )
%% ��������      
P_percent1 = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size      
P_percent2 = 0.4;
P_percent3 = 0.65;
pNum1 = round( N *  P_percent1 );    % The population size of the producers   
pNum2 = round( N *  P_percent2 );
pNum3 = round( N *  P_percent3 );

%% ��ʼ����Ⱥ��ʹ�û�������Bernoulli��
%Initialization
Z = rand(N, dim);   % �������һ��dά����
lambda = 0.4;
for i = 1 : N  
     for j=1:dim
       if (Z(i,j)<=(1-lambda)) && (Z(i,j)>0)
          Z(i,j) = Z(i,j)/(1-lambda);
          
       else 
          Z(i,j)=(Z(i,j)-1+lambda)/lambda;
       end  
     end
end

% ��z�ĸ��������ز�����Ӧ������ȡֵ����
x = zeros(N, dim);
for i= 1 : N
    x(i,:) = lb + (ub - lb) .* Z(i,:);
    fit( i ) = fobj( x( i, : ) ) ;   
end

%% ����ǰ׼��
pFit = fit;                       
pX = x; 
XX=pX;    
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX ȫ�����λ��

 %% ��ʼ����
 % Start updating the solutions.
for t = 1 : Max_iter    
 %% ��MSA�Ľ��������㷨Ƕ��Equation  
   Wmax = 0.9;
   Wmin = 0.782;
   r1 = (Wmax - Wmin)/2 * cos(pi*t/Max_iter) + (Wmax + Wmin)/2; 
   w = Wmax - (Wmax - Wmin) * (t/Max_iter);
    
       [~,B]=max(fit);
        worse= x(B,:);   
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : pNum1    
       if(rand(1) < 0.5)
          b=0.3;   % bΪ0��1
          k=0.1;   % kΪ0��0.2
          r4=rand(1);
          miu=0.1;
          if (r4>miu)
           a=1;
          else
           a=-1;
          end
    x( i , : ) =  pX(  i , :)+b*abs(pX(i , : )-worse)+a*k*(XX( i , :)); % Equation (1)����
       else
    for j = 1:dim 
            % ��Eq. (3.3)����r2,r3,r4
            r2 = (2*pi)*rand();
            r3 = 2*rand;
            % Eq. (3.3)
         x(i, j) = w * x(i, j)+(r1*sin(r2) * (r3*bestX(j)-x(i, j))); 
     end
       end
        x(  i , : ) = Bounds( x(i , : ), lb, ub );    %�߽�Լ��
        fit(  i  ) = fobj( x(i , : ) ); 
    end
    
   % �������  
 [ ~, bestII ] = min( fit );      % fMin denotes the current optimum fitness value
  bestXX = x( bestII, : );             % bestXX ����ǰ��� 

 R=1-t/Max_iter;                           %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Xnew1 = bestXX.*(1-R); 
     Xnew2 =bestXX.*(1+R);                    %%% Equation (3)
   Xnew1= Bounds( Xnew1, lb, ub );
   Xnew2 = Bounds( Xnew2, lb, ub );
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   Xnew11 = bestX.*(1-R); 
   Xnew22 =bestX.*(1+R);                     %%% Equation (5)
   lbnew2= Bounds( Xnew11, lb, ub );
   ubnew2 = Bounds( Xnew22, lb, ub );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    for i = ( pNum1 + 1 ) : pNum2                  % Equation (4)
     x( i, : )=bestXX+((rand(1,dim)).*(pX( i , : )-Xnew1)+(rand(1,dim)).*(pX( i , : )-Xnew2));
   x(i, : ) = Bounds( x(i, : ), Xnew1, Xnew2 );
  fit(i ) = fobj(  x(i,:) ) ;
    end
    
 for i = (pNum2+1): pNum3                  %  small-dung beetles Equation (6)
     
       x( i, : )=  pX( i , : )+((randn(1)).*(pX( i , : )-lbnew2)+((rand(1,dim)).*(pX( i , : )-ubnew2)));
       x(i, : ) = Bounds( x(i, : ),lb, ub);
       fit(i ) = fobj(  x(i,:) ) ;
  
 end
  
 
 
  for j = (pNum3+1) : N     % Equation (7)
      
      s=1;
       x( j,: )=bestX+s*randn(1,dim).*((abs(( pX(j,:  )-bestXX)))+(abs(( pX(j,:  )-bestX))))./2;
      x(j, : ) = Bounds( x(j, : ), lb, ub );
      fit(j ) = fobj(  x(j,:) ) ;
  end  

   %% Update the individual's best fitness vlaue and the global best fitness value
     XX=pX;
    for i = 1 : N 
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
    
        
   %% ����Ӧ��˹-�����Ŷ�����
      w1=t/Max_iter;
      w2=1-t/Max_iter;
      x(i,:) = pX(i,:)*(1 + w1 * randn + w2 * tan((rand-1/2)*pi));   %��˹-�����Ŷ�����
      x(i, : ) = Bounds( x(i, : ), lb, ub ); %�߽�
      fit(i ) = fobj(  x(i,:) ) ;
      
       if ( fit( i ) < pFit( i ) )   %��������
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
       end  
        
         
        if( pFit( i ) < fMin )
           % fMin= pFit( i );
           fMin= pFit( i );
            bestX = pX( i, : );
          %  a(i)=fMin;
            
        end
    end
  
     Convergence_curve(t)=fMin;
  disp(['MSADBO: At iteration ', num2str(t), ' ,the best fitness is ', num2str(Convergence_curve(t))]);
    
end 
end

% Application of simple limits/bounds

%---------------------------------------------------------------------------------------------------------------------------
function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
end

function S = Boundss( SS, LLb, UUb)
  % Apply the lower bound vector
  temp = SS;
  I = temp < LLb;
  temp(I) = LLb(I);
  
  % Apply the upper bound vector 
  J = temp > UUb;
  temp(J) = UUb(J);
  % Update this new move 
  S = temp;
end