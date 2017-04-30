function Thetas = unpackThetas(packed, layersizes, varargin)
% [Theta1, Theta2, ...] = packThetas(packedthetas, layersizes)
% layersizes = [s1, ..., sn], where 1=unput, 2..n-1=hidden, n=output layer
% return varargout = {Theta1, Theta2, ..., Theta{n-1}}
  
  layersizes = [layersizes varargin{:}];
  % # of Thetas is # of layers minus one
  n_Thetas = length(layersizes) - 1;
  
  % Theta{i} is a matrix of size (r x c) = (s[i+1]) x (s[i]+1)
  rows = layersizes(2:end);
  columns = layersizes(1:end-1)+1;
  numels = rows.*columns;
  % start idx of each Theta in the packed vector
  starts = 1+[0 cumsum(numels(1:end-1))];
  ends = starts+numels-1;
  
  Thetas = cell(1, n_Thetas);
  for i = 1:n_Thetas
    Thetas{i} = reshape(packed(starts(i):ends(i)), ...
                          rows(i), columns(i));
  end %for
end
 
 %{
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%}