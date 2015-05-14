function net = cnnff(net, x)
%CNNFF Performs a feed-forward evaluation of a CNN.
%
%  This function can evaluate the CNN over multiple images simultaneously.
%  All of the key operations are performed using the 'convn' function,
%  which can apply the specified convolution to multiple images in one
%  call.
%
%  Paramaters
%    net  - Structure containing all of the parameters of a fully trained
%           CNN.
%      x  - Inputs to evaluate the CNN over.
%           The inputs should be formated as images.
%           For example, the MNIST test examples are presented as a
%           28 x 28 x 10,000 matrix.
%
%  Returns
%    net  - CNN with all of its output variables populated.
%      net.layers{1}.a{j} - Output map 'j' for layer 'l'.
%      net.fv - Final feature vectors for all input examples.
%      net.o - Output of the final perceptrons for all input examples.

    % 'n' is the number of layers in the network, including the input
    % layer.
    n = numel(net.layers);
    
    % 'a' is used to hold the output mpas for a layer. It is a cell array 
    % with one entry per output map.
    % The output of the first layer is just the input vectors, x.
    net.layers{1}.a{1} = x;
    
    % This variable will hold the number input maps to the current layer,
    % which is just the number of output maps (or "features") in the 
    % previous layer.
    % We start with layer 2, so there is only one input map.
    inputmaps = 1;

    % For each layer in the network (skipping over the input layer)...
    for l = 2 : n
        
        % For convolutional layers...
        if strcmp(net.layers{l}.type, 'c')
            % For each of this layer's output maps (that is, for each
            % filter / feature in this layer)...
            for j = 1 : net.layers{l}.outputmaps
                
                % Create blank matrix z to hold the output map (feature 
                % activations) for feature 'j'.
                % The size of the output map is equal to the size of the 
                % previous layer's output, but reduced by the width of the
                % filter, since the filter can't slide past the edge of the
                % image.
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1, net.layers{l}.kernelsize - 1, 0]);
                
                % For each input map (that is, for each output map in the 
                % previous layer...)
                for i = 1 : inputmaps
                    % Select output map 'i' from the previous layer, and
                    % apply filter 'j' to it with a matrix convolution.
                    % We'll keep a running sum of the convolutions over
                    % each of the input maps.
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                
                % Add the bias term, and apply the sigmoid function.
                % Store the result as outputmap 'j' of layer 'l'.
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            % Set the number of input maps for the next layer.
            inputmaps = net.layers{l}.outputmaps;
        
        % For subsampling (pooling) layers...
        elseif strcmp(net.layers{l}.type, 's')
            % For each input map...
            for j = 1 : inputmaps
                % Subsample the input map by averaging the values.
                % For example, to subsample an image by a factor of two,
                % first, compute the averages by convolving it with the 
                % filter mask:
                %    [1/4, 1/4;
                %     1/4, 1/4];
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   
                
                % Then take only every 'scale'th pixel.
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    % Concatenate all end layer feature maps into vector
    net.fv = [];
    
    % For each of the output maps in the final layer...
    for j = 1 : numel(net.layers{n}.a)
        % Get the size of output map 'j' for the final layer.
        sa = size(net.layers{n}.a{j});
        
        % Unwind output map 'j' into a vector, and concatenate it with
        % the feature vector. sa(1) and sa(2) are the dimensions of the 
        % output map, and sa(3) corresponds to the number of images in
        % the input matrix 'x'.
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    % Feedforward into output perceptrons
    % Multiply the feature vector by the output weights and add the bias 
    % term.
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

end
