tic
load('compiled_probswitch_data_09122020.mat','data')
iter_results = [];
for i = 1
    [results] = fit_models(data,3);
    %iter_results(i).HMM = results(1);
    %iter_results(i).SRL = results(2);
    %iter_results(i).BRL = results(3);
    BRL_results = results(1);
    save(sprintf('BRL_modeling_output_%d',i),'BRL_results')
end    
%save('all_modeling_output','iter_results')
toc