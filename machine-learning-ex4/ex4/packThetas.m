function packedthetas = packThetas(varargin)
# packed = packThetas(Theta1, Theta2, ...)
  packedthetas = [];
  for i=1:nargin
    Theta = varargin{i};
    packedthetas = [packedthetas; Theta(:)];
  end
 end