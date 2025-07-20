function [output_element] = str_split_mw(str, d, element)

    filename = split(str, d); 
    
    if nargin < 3 
        % Default to last element
        element = numel(filename);
    end

    % If pulling the last element == 0 
    if element == 0 
        output_element = filename(end);
        return
    end
    
    if any(element < 0)
        for j = 1:numel(element) 
            out(j) = filename(end-abs(element(j)));
        end   
%         output_element = strjoin(out, '_');
        output_element = out;
    end
    
    % If only pulling a single element
    if element ~= 0
        if numel(element) == 1
            for i = 1:length(filename)
                if element == i 
                    output_element = filename{i};
                end
            end
        end
    end

    % if pulling elements in order isInOrder = all(diff(element) == 1);
    if element ~= 0
        if numel(element) > 1 
            for i = 1:length(filename)
                if any(ismember(element,i))
                    output_element{i} = filename{i};
                end
            end

            % Find non-empty cells
            nonEmptyIndices = ~cellfun(@isempty, output_element);

            % Keep only non-empty cells
            output_element = output_element(nonEmptyIndices);
            output_element = strjoin(output_element, d);
        end
    end

end

