clc
clear all
close all



k = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
for i=1:26
    v{i} = 28+i;
end

map = containers.Map(k,v)

labels_real = {'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'O', 'ZERO'}

for i=1:11
    curr_label_real = labels_real{i}
    curr_label_num = []
    for ii=1:length(curr_label_real)
        curr_label_num(ii) = map(curr_label_real(ii))
    end
    labels_num{i} = curr_label_num
end

