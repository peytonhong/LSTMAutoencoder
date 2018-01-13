% plot comparison

data = csvread('auto_save_profiles_sample_predictor.txt');
seq_length = data(:,1);
hidden_size = data(:,2);
loss = data(:,3);

max_seq = max(data(:,1));
max_hid = max(data(:,2));
mesh_data = ones(max_seq, max_hid);

for i=1:size(data,1)
    seq = data(i,1);
    hid = data(i,2);
    loss = data(i,3);
    mesh_data(seq, hid) = loss;
end


imagesc(mesh_data, [0, 0.005])
xlabel('hidden size')
ylabel('sequence length')
zlabel('loss')
view([-90,90])