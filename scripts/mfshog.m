N= 20;

namesSa = {};
namesL = {};

ult = 0;

baguette = [];
salvado = [];
lactal = [];
sandwich = [];

baguetteC = [];
salvadoC = [];
lactalC = [];
sandwichC = [];

imagenes3 = [];
direc = '/home/rodrigo/rodrigo/europeanfood/images/nonbread/res/';
archivos = dir(direc);
for i = 3:size(archivos,1),
    namesIm = '';
    namesIm = strcat(direc,archivos(i).name);
    if(size(imread(namesIm),3) == 3) % if it is RGB
        imagenes3 = [imagenes3; LapMFS(namesIm)];
    end
end


    for i = 1:20,       
        namesB{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/scanner/baguette/baguette',int2str(i+ult),'.tif');
        namesL{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/scanner/lactal/lactal',int2str(i+ult),'.tif');
        namesS{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/scanner/salvado/salvado',int2str(i+ult),'.tif');
        namesSa{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/scanner/sandwich/sandwich',int2str(i+ult),'.tif');
        
        namesBC{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/camera/baguette/b',int2str(i+ult),'.tif');
        namesLC{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/camera/lactal/l',int2str(i+ult),'.tif');
        namesSC{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/camera/salvado/s',int2str(i+ult),'.tif');
        namesSaC{i} = strcat('/home/rodrigo/rodrigo/mecom2012/mecom/imagenes/camera/sandwich/s',int2str(i+ult),'.tif');
    end

    %sandwich = [];
    for i = 1:20
        baguette = [baguette; LapMFS(namesB{i})];
        lactal = [lactal; LapMFS(namesL{i})];
        salvado = [salvado; LapMFS(namesS{i})];
        sandwich = [sandwich; LapMFS(namesSa{i})];
        
        
        baguetteC = [baguetteC; LapMFS(namesBC{i})];
        lactalC = [lactalC; LapMFS(namesLC{i})];
        salvadoC = [salvadoC; LapMFS(namesSC{i})];
        sandwichC = [sandwichC; LapMFS(namesSaC{i})];
        
    end
    
    
    fractal2otroS = [baguette; lactal; salvado; sandwich; imagenes3(1:20,:)];
    fractal2otroC = [baguetteC; lactalC; salvadoC; sandwichC; imagenes3(21:40,:)];

    csvwrite('fractal2otroS.csv',fractal2otroS);
    csvwrite('fractal2otroC.csv',fractal2otroC);
    
    %libsvmwrite('fractal2otroS.txt',labels',sparse(fractal2otroS));
    %libsvmwrite('fractal2otroC.txt',labels',sparse(fractal2otroC));
    
    %fractal2s = csvread('/home/rodrigo/rodrigo/mecom2012/mecom/exps/fractal2s.csv');
    %fractal2c = csvread('/home/rodrigo/rodrigo/mecom2012/mecom/exps/fractal2c.csv');

    %fractal2Es = [fractal2s fractal2otroS];
    %fractal2Ec = [fractal2c fractal2otroC];

    %csvwrite('fractal2Es.csv',fractal2Es);
    %csvwrite('fractal2Ec.csv',fractal2Ec);

    %libsvmwrite('fractal2Es.txt',labels',sparse(fractal2Es));
    %libsvmwrite('fractal2Ec.txt',labels',sparse(fractal2Ec));
    

