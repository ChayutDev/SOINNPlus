% GSOINN+ - Self-Organizing Incremental Neural Network with Ghost Node
% ver. 1.1.0
%
% Copyright (c) 2019 Chayut Wiwatcharakoses
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php

classdef GSoinnPlus < SoinnPlus

    properties

        % GSOINN+ parameter
        fracParam;
        fracFlag;
        limit;

        % The label for supervise training
        labels;
        
        % Ghost Node parameter
        subSpace;
        subLabel;
        hasSubspace;
        distanceHandler;
        
    end

    methods
        function obj = GSoinnPlus(varargin)
            % Call SoinnPlus' Constructor
            obj@SoinnPlus(varargin)
            
            % GSOINN+'s constructor
            o = OptionHandler(varargin);
            obj.fracParam = o.get('FracPow', 2);
            obj.limit = o.get('limit', Inf);
            obj.maxEdgeAge = o.get('ageMax', 50);
            obj.fracFlag = o.get('fractional', 0);
            
            obj.labels = [];
            obj.hasSubspace = logical([]);
            obj.subSpace = {};
            obj.subLabel = {};

            % Option for selecting Euclidean distance or fractional
            % distance
            if obj.fracFlag == 0
                obj.distanceHandler = @(a,b,c) sum(((b - repmat(a, size(b, 1), 1)).^2), 2);
            else
                obj.distanceHandler = @obj.fracDist;
            end

        end

        % Override SoinnPlus function
        function inputSignal(obj, signal, varargin)
            % @param {row vector} signal - new input signal
            signal = obj.checkSignal(signal);
            obj.signalNum = obj.signalNum + 1;
            
            % Check if the label is given
            if isempty(varargin)
                label = 0;
            else
                label = varargin{1};
            end
            
            % If in initialization state add node unconditionally
            if size(obj.nodes, 1) < 3 
                obj.addNode(signal, label);
                return;
            end
            
            % Find the winners and calculate similarity threshold
            [winner, dists] = obj.findNearestNodes(2,signal);
            simThresholds = obj.calculateSimiralityThresholds(winner);

            % Check if the network should create the link between both
            % winner or not
            if any(obj.runThVariance == 0) 
                eFlag = 1; % Create link unconditionally if there is no edge in the network
            else
                % Check if edge between both winner should be created or
                % not
                th = obj.paramC*sqrt(obj.runThVariance/obj.linkCreated);
                noises = sum(obj.adjacencyMat > 0) < 1; 
                data = ~full(noises);
                trustLv = obj.winningTimes(winner)./max(obj.winningTimes(data)); % Calculate trustworthiness
                eFlag = any(sqrt(simThresholds).*(1-trustLv') < (obj.runThMean + th)');
            end
       
            % Add node if either one of distance greater than corresponding similarity threshold
            if any(dists > simThresholds) 
                obj.addNode(signal, label);
            else
                
                % Add edge if the the condition is true
                if eFlag
                    isNew = obj.addEdge(winner);
                    if isNew
                        obj.linkCreated = obj.linkCreated+1;
                        preMean = obj.runThMean;
                        % Update the mean and variance of similarity
                        % threhsold
                        obj.runThMean = obj.runThMean + (sqrt(simThresholds') - obj.runThMean)./obj.linkCreated;
                        obj.runThVariance = obj.runThVariance + (sqrt(simThresholds') - preMean).*(sqrt(simThresholds') - obj.runThMean);
                    end
                end
                
                % After process updating
                obj.incrementEdgeAges(winner(1));
                winner(1) = obj.deleteEdgeHandler(winner(1));
                obj.updateWinner(winner(1), signal, varargin{1});
                obj.updateAdjacentNodes(winner(1), signal);
            end

            % Check if any node can be deleted
            if mod(obj.signalNum, obj.deleteNodePeriod) == 0
                obj.deleteNoiseHandler();
            end
        end
        
        % Function for classify the input cases
        function [pred, labels, score] = classify(obj, X, K)
            % Search for the K nearest nodes for each input case
            [index ,dist] = knnsearch(obj.nodes, double(X), 'K', K);
            class = obj.labels';
            labels = class(index);

            % Check if any of them has ghost node
            for i = 1:size(X,1)    
                for j = 1:K
                    if obj.hasSubspace(index(i,j))
                        [idx, dist(i,j)] = knnsearch(obj.subSpace{index(i,j)}, double(X(i,:)), 'K', 1);
                        labels(i,j) = obj.subLabel{index(i,j)}(idx);
                    end
                end
            end
            
            % Score calculation
            score = zeros(size(X,1), max(obj.labels));
            classW = zeros(size(X,1), max(obj.labels));
            for i = 1:size(X,1)
                for k = 1:K
                    % using inverse Euclidean distance
                    classW(i,labels(i,k)) = classW(i,labels(i,k)) + (dist(i, k).^-1);
                end
                
                for c = 1:max(obj.labels)
                    score(i,c) = (classW(i,c) / sum(classW(i,:)));
                end
                
            end
                
            % Evaluate
            [~, pred] = max(score, [], 2); 
        end
        
        % Minkowsi Formula (Lp norm)
        function d = fracDist(~, x, a, p)
            d = sum(abs(bsxfun(@minus, a,x)).^p, 2).^(1/p);
        end
    end
%%
    methods(Hidden=true)
     
        function addNode(obj, signal, varargin)
            num = size(obj.nodes, 1);
            obj.nodes(num+1,:) = signal;
            obj.winningTimes(num+1) = 1;
            obj.hasSubspace(num+1) = false;
            obj.winTS(num+1) = obj.signalNum;
            obj.nodeTS(num+1) = obj.signalNum;
            obj.labels(num+1) = varargin{1};
            obj.subSpace{num+1,1} = [];
            obj.subLabel{num+1,1} = [];

            if num == 0
                obj.adjacencyMat(1,1) = 0;
            else
                obj.adjacencyMat(num+1,:) = zeros(1, num);
                obj.adjacencyMat(:,num+1) = zeros(num+1, 1);
            end
        end
        
        function [indexes, sqDists] = findNearestNodes(obj, num, signal)
            indexes = zeros(num, 1);
            sqDists = zeros(num, 1);

            D = obj.distanceHandler(signal, obj.nodes, obj.fracParam);
            for i = 1:num
                [sqDists(i), indexes(i)] = min(D);
                D(indexes(i)) = inf;
            end
        end
        

        function simThresholds = calculateSimiralityThresholds(obj, nodeIndexes)
            simThresholds = zeros(length(nodeIndexes), 1);
            for i = 1: length(nodeIndexes)
                simThresholds(i) = obj.calculateSimiralityThreshold(nodeIndexes(i));
            end
        end

        function threshold = calculateSimiralityThreshold(obj, nodeIndex)
            if any(obj.adjacencyMat(:,nodeIndex))
                pals = obj.nodes(obj.adjacencyMat(:,nodeIndex) > 0,:);
                D = obj.distanceHandler(obj.nodes(nodeIndex,:), pals, obj.fracParam);
                threshold = max(D);
            else
                [~, sqDists] = obj.findNearestNodes(2, obj.nodes(nodeIndex, :));
                threshold = sqDists(2);
            end
        end

        function isNew = addEdge(obj, nodeIndexes)
            if obj.adjacencyMat(nodeIndexes(1), nodeIndexes(2)) || obj.adjacencyMat(nodeIndexes(2), nodeIndexes(1))
                isNew = true;
            else
                isNew = false;
            end
            obj.adjacencyMat(nodeIndexes(1), nodeIndexes(2)) = 1;
            obj.adjacencyMat(nodeIndexes(2), nodeIndexes(1)) = 1;
        end

        function updateWinner(obj, winnerIndex, signal, varargin)
            % @param {int} winnerIndex - hte index of winner
            % @param {row vector} signal - inputted new signal
            obj.winningTimes(winnerIndex) = obj.winningTimes(winnerIndex) + 1;
            w = obj.nodes(winnerIndex,:);
            obj.nodes(winnerIndex, :) = w + (signal - w)./obj.winningTimes(winnerIndex);
            obj.winTS(winnerIndex) = obj.signalNum;
            
            % Check Label for subspace
            if obj.labels(winnerIndex) ~= varargin{1}
                obj.hasSubspace(winnerIndex) = true;
                if isempty(obj.subSpace{winnerIndex})
                    obj.subSpace{winnerIndex} = [w; signal];
                    obj.subLabel{winnerIndex} = [obj.labels(winnerIndex); varargin{1}];
                else
                    obj.subSpace{winnerIndex} = [obj.subSpace{winnerIndex}; signal];
                    obj.subLabel{winnerIndex} = [obj.subLabel{winnerIndex}; varargin{1}];
                end
            end
        end

        function hasNei = updateAdjacentNodes(obj, winnerIndex, signal)
            pals = find(obj.adjacencyMat(:,winnerIndex) > 0);
            hasNei = any(pals);
            for i = 1:length(pals)
                w = obj.nodes(pals(i),:);
                obj.nodes(pals(i), :) = w + (signal - w)./(100 * obj.winningTimes(pals(i)));
            end
        end

        function deleteNodes(obj, indexes)
            % record deleted node's signatures and remove signature
            % recoders.
            
            % remove the nodes.
            obj.nodes(indexes,:) = [];
            obj.winningTimes(indexes) = [];
            obj.adjacencyMat(indexes, :) = [];
            obj.adjacencyMat(:, indexes) = [];
            obj.winTS(indexes) = [];
            obj.nodeTS(indexes) = [];
            obj.labels(indexes) = [];
            obj.hasSubspace(indexes) = [];
            obj.subSpace(indexes) = [];
            obj.subLabel(indexes) = [];
            obj.nodeDeleted = obj.nodeDeleted + sum(indexes);
        end

    end
end