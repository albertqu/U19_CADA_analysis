%modeling_output = load('results.mat');
load('final_modeling_output_092620.mat');
%modeling_output =load('G_test.mat');
input = load('compiled_probswitch_data_09122020.mat');

comp_data = input.data;

RAW = cell(1,length(comp_data));
for j = 1:length(comp_data)
	sesh = unique(comp_data(j).s);
	for i = 1:length(sesh)
        RAW{j}(i).a = comp_data(j).a(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'));
        RAW{j}(i).r = comp_data(j).r(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'));
        RAW{j}(i).c = comp_data(j).c(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'));
        RAW{j}(i).s = comp_data(j).s(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'));
	end    
end

%modeling_data = modeling_output.results(3).latents;
modeling_data = iter_results.BRL.latents;
BRL = [];
for j = 1:length(comp_data)
	sesh = unique(comp_data(j).s);
	for i = 1:length(sesh)
        BRL{j}(i,:).b = modeling_data(j).b(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
        BRL{j}(i,:,:).w = modeling_data(j).w(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:,:);
        BRL{j}(i,:).q = modeling_data(j).q(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
        BRL{j}(i,:).acc = modeling_data(j).acc(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
        BRL{j}(i,:).rpe = modeling_data(j).rpe(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
	end    
end

%modeling_data = modeling_output.results(2).latents;
modeling_data = iter_results.SRL.latents;
SRL = [];
for j = 1:length(comp_data)
	sesh = unique(comp_data(j).s);
	for i = 1:length(sesh)
        SRL{j}(i,:).q = modeling_data(j).q(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
        SRL{j}(i,:).acc = modeling_data(j).acc(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
        SRL{j}(i,:).rpe = modeling_data(j).rpe(find(comp_data(j).s==sesh(i),1,'first'):find(comp_data(j).s==sesh(i),1,'last'),:);
	end    
end
save('probswitch_modeling_output','BRL','SRL','RAW')

% BSD002: modeling_code = [11,12,13,16,21,24,27,32,33,37,42]; i.e. the
%         ages = {'p151_session1','p151_session2','p153','p156','p232','p235','p238','p243','p244','p248','p252'};
% first day of FP is the 11th session in modeling code, 4th FP day is 16th,
% etc.
          
% BSD003: modeling_code = [3,4,7,10,13,17,24,28,32,35,42];
%         ages = {'p147_LH','p147_RH','p221','p224','p227','p231','p238','p242','p246','p249','p256'};
% BDS004: modeling_code = [7,8,13,15,26,30,34,38,41,48];
%         ages = {'p145','p146','p220','p222','p233','p237','p241','p245','p248','p255'};
% BDS005: modeling_code = [9,10,11,12,23,29,33,37,40,43,47,50];
%         ages = {'p102','p103','p104','p105','p189','p195','p199','p203','p206','p209','p213','p216'};
% BSD006: modeling_code = [26,43];
%         ages = {'p140','p159'};
% BSD007: modeling_code = [25,29,34,43];
%         ages = {'p139','p143','p148','p157'};
% BSD008: modeling_code = [24,28,33,38,41,45];
%         ages = {'p138','p142','p147','p152','p156','p161'};
% BSD009: modeling_code = [24,25,29,34,37,42,46];
%         ages = {'p135_session1','p135_session2','p140','p144','p148','p153','p158'};