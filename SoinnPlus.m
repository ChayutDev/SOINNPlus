% SOINN+ - Self-Organizing Incremental Neural Network Plus
% ver. 1.0.1
%
% Copyright (c) 2016 Yoshihiro Nakamura
% Copyright (c) 2019 Chayut Wiwatcharakoses
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php

classdef SoinnPlus < handle

    properties
        % Internal variable.
        nodeDelTh;
        nodeAvgIdleDel;
        edgeAvgLtDel;
        curNodeTh;
        curEdgeTh;
		enableTracking = 1;
        minDegree = 1;
        paramEdge = 2;
        paramC = 2;
        paramAlpha = 2;
        
        % Parameter
        dimension;
        deleteNodePeriod;
        maxEdgeAge;
        
        % SOINN+ Option
        nodeFlag;
        edgeFlag;

        % Data related variable
        signalNum;
        nodes;
		trackInput;
		trackInputIdx;
        winningTimes;
        winTS;
        nodeTS;
        adjacencyMat;
        linkCreated;
        nodeDeleted;
        edgeDeleted;
        runThVariance;
        runThMean;

        % Function Handler
        deleteNoiseHandler;
        deleteEdgeHandler;
       
        
    end

    methods
        function obj = SoinnPlus(varargin)
            % constractor
            % Legacy Options: 
            % 'lambda' (default: 300)
            %       A period deleting nodes. The nodes that doesn't satisfy
            %       some condition are deleted every this period.
            % 'ageMax' (default: 50)
            %       The maximum of edges' ages. If an edge's age is more
            %       than this, the edge is deleted.
            % 'dim' (default: 2)
            %       signal's dimension
            %
            % Plus Options:
            % 'node' (default: 1)
            %       1 - Enable plus version of node deletion
            %       0 - Disable plus version of node deletion
            % 'edge' (default: 1)
            %       1 - Enable plus version of node linking
            %       0 - Disable plus version of node linking
            
            o = OptionHandler(varargin);
            obj.deleteNodePeriod = o.get('lambda', 300);
            obj.maxEdgeAge = o.get('ageMax', 50);
            obj.dimension = o.get('dim', 2);
            
            obj.nodeFlag = o.get('node', 1);  
            obj.edgeFlag = o.get('edge', 1);  
            obj.runThVariance = [0 0];
            obj.runThMean = [0 0];
            obj.linkCreated = 0;
            obj.nodes = [];
			obj.trackInput = {};
			obj.trackInputIdx = {};
            obj.winningTimes = [];
            obj.winTS = [];
            obj.nodeTS = [];
            obj.nodeDelTh = 0;
            obj.nodeAvgIdleDel = 0;
            obj.edgeAvgLtDel = 0;
            obj.adjacencyMat = sparse([]);
            obj.signalNum = 0;
            obj.nodeDeleted = 0;
            obj.edgeDeleted = 0;
 
            if obj.nodeFlag == 0
                obj.deleteNoiseHandler = @obj.deleteNoiseNodes_Original;
            else
                obj.deleteNoiseHandler = @obj.deleteNoiseNodes_Plus;
                obj.deleteNodePeriod = 1;
            end
            
            if obj.edgeFlag == 0
                obj.deleteEdgeHandler = @obj.deleteOldEdges_Original;
            else
                obj.deleteEdgeHandler = @obj.deleteOldEdges_Plus;
            end

        end

        % Main function when receiving new cases goes here ---------------
        function inputSignal(obj, signal, varargin)
            % @param {row vector} signal - new input signal
            signal = obj.checkSignal(signal);
            obj.signalNum = obj.signalNum + 1;

            % If in initialization state add node unconditionally
            if size(obj.nodes, 1) < 3 
                obj.addNode(signal, 0);
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
                obj.addNode(signal, 0);
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
                obj.updateWinner(winner(1), signal);
                obj.updateAdjacentNodes(winner(1), signal);
            
			end

            % Check if any node can be deleted
            if mod(obj.signalNum, obj.deleteNodePeriod) == 0
                obj.deleteNoiseHandler();
            end
        end

        function show(obj, varargin)
            % Display SOINN's network in 2D.
            % Options:
            % 'winningTimes' a flag to show winning time of each node
            % 'dim' which dimensions to show (default: [1, 2])
            % 'data' data showed with KDESOINN status (default: [])
            % 'cluster' cluster labels which is a row vector outputed by clustring()
            o = OptionHandler(varargin);
            dims = o.get('dim', [1 2]);
            winningTimeFlag = o.exist('winningTimes');
            data = o.get('data', []);
            hold on;
            %show data
            if size(data, 1) > 0
                plot(data(:,dims(1)), data(:,dims(2)), 'xb');
            end
            % show edges
            for j = 1:size(obj.adjacencyMat,2)
                for k = j:size(obj.adjacencyMat, 1)
                    if obj.adjacencyMat(k,j) > 0
                        nk = obj.nodes(k,:);
                        nj = obj.nodes(j,:);
                        plot([nk(dims(1)), nj(dims(1))], [nk(dims(2)), nj(dims(2))], 'k');
                    end
                end
            end
            % show nodes
            clusterLabels = o.get('cluster', []);
            if ~isempty(clusterLabels)
                clusterNum = max(clusterLabels);
                colors = obj.getRgbVectors(clusterNum+1);
                idx = clusterLabels == -1;
                plot(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2)), '.', 'Markersize', 20, 'MarkerEdgeColor', colors(1, :));
                for i = 1:clusterNum
                    idx = clusterLabels == i;
                    plot(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2)), '.', 'Markersize', 20, 'MarkerEdgeColor', colors(i+1, :));
                    %text(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2))-5, num2str(obj.winningTimes(idx))); %'\leftarrow sin(\pi)')
                end
            else
                plot(obj.nodes(:,dims(1)), obj.nodes(:,dims(2)), '.b','Markersize',20);
               % plot(obj.delNodes(:,dims(1)), obj.delNodes(:,dims(2)), '.r','Markersize',15);
            end
            
            % plot Winning Time
            %for i = 1:size(obj.nodes,1)
            %    text(obj.nodes(i, dims(1))+1, obj.nodes(i, dims(2)), num2str(obj.winningTimes(i))); %'\leftarrow sin(\pi)')
            %end
            
            % show winningTimes of nodes
            if winningTimeFlag
                delta = (max(max(obj.nodes(:,dims))) - min(min(obj.nodes(:,dims)))) * 0.005;
                for i = 1:size(obj.nodes,1)
                    text(obj.nodes(i,dims(1)) + delta, obj.nodes(i,dims(2)) + delta, num2str(obj.winningTimes(i)), 'BackgroundColor', [1,1,1]);
                end
            end
%             title(strcat('nodes:', num2str(size(obj.nodes,1)), ' edges:', num2str(sum(sum((obj.adjacencyMat > 0)))), ' WinTime:', num2str(sum(obj.winningTimes)) ));
           
            %title(strcat('nodes:', num2str(size(obj.nodes,1)), ' edges:', num2str(sum(sum((obj.adjacencyMat > 0)))), ' WinTime:', num2str(sum(obj.winningTimes)) , ' DN:', num2str(obj.nodeDeleted), ' DE:', num2str(obj.edgeDeleted)));
            set(gca,'XGrid','on','YGrid','on');
            hold off
            if o.get('save', false)
                saveas(gcf, o.get('savePath', ['tmp.png']))
            end
        end

        function err = topoErr(obj, Data)
            numErr = 0;
            for i = 1:size(Data, 1)
                nearIdx = obj.findNearestNodes(2, Data(i,:));
                if obj.adjacencyMat(nearIdx(1), nearIdx(2))
                    numErr = numErr + 1;
                end
            end
            err = numErr/size(Data,1);
        end
    end
%%
    methods(Hidden=true)
        function rgbs = getRgbVectors(obj, num)
            rgbs = [];
            maxBase = ceil(power(num, 1/3));
            for base = 2:maxBase
                candi = zeros(base^3, 3);
                for i = 0:base^3-1
                    candi(i+1,:) = obj.str2numArry(dec2base(i,base,3));
                end
                candi = candi / (base-1);
                for i = 1:size(candi, 1)
                    tf = true;
                    for k = 1:size(rgbs, 1)
                        if all(candi(i,:) == rgbs(k,:))
                            tf = false;
                            break;
                        end
                    end
                    if tf
                        rgbs = cat(1, rgbs, candi(i, :));
                    end
                end
            end
            rgbs = rgbs(1:num, :);
        end

        function arry = str2numArry(~, str)
            n = length(str);
            arry = zeros(1, n);
            for i = 1:n
                arry(i) = str2num(str(i));
            end
        end

        function [labels, num] = labelWithBreadthFirstSearch(obj, labels, queue, currentClusterLabel)
            num = 0;
            while ~isempty(queue)
                idx = queue(1);
                queue(1) = [];
                if labels(idx) == 0
                    num = num + 1;
                    labels(idx) = currentClusterLabel;
                    queue = cat(2, queue, find(obj.adjacencyMat(idx, :) > 0));
                end
            end
        end

        function signal = checkSignal(obj, signal)
            s = size(signal);
            if s(1) == 1
                if s(2) == obj.dimension
                    return;
                else
                    error('Soinn:checkSignal:dimError', 'The dimension of input signal is not valid.');
                end
            else
                if s(2) == 1
                    signal = obj.checkSignal(signal');
                else
                    error('Soinn:checkSignal:notVector', 'Input signal have to be a vector.');
                end
            end
        end

        function addNode(obj, signal, varargin)
            num = size(obj.nodes, 1);
            obj.nodes(num+1,:) = signal;
            obj.winningTimes(num+1) = 1;
            obj.winTS(num+1) = obj.signalNum;
            obj.nodeTS(num+1) = obj.signalNum;

            if num == 0
                obj.adjacencyMat(1,1) = 0;
            else
                obj.adjacencyMat(num+1,:) = zeros(1, num);
                obj.adjacencyMat(:,num+1) = zeros(num+1, 1);
            end
			
			if obj.enableTracking
				obj.trackInput{num+1} = signal;
				obj.trackInputIdx{num+1} = obj.signalNum;
			end
        end

        function [indexes, sqDists] = findNearestNodes(obj, num, signal)
            indexes = zeros(num, 1);
            sqDists = zeros(num, 1);
            %D = sum(((obj.nodes - repmat(signal, size(obj.nodes, 1), 1)).^2), 2);
            D = sum((bsxfun(@minus,obj.nodes,signal).^2), 2);
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
                D = sum(((pals - repmat(obj.nodes(nodeIndex,:), size(pals, 1), 1)).^2), 2);
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

        function updateWinner(obj, winnerIndex, signal)
            % @param {int} winnerIndex - hte index of winner
            % @param {row vector} signal - inputted new signal
            obj.winningTimes(winnerIndex) = obj.winningTimes(winnerIndex) + 1;
            w = obj.nodes(winnerIndex,:);
            obj.nodes(winnerIndex, :) = w + (signal - w)./obj.winningTimes(winnerIndex);
            obj.winTS(winnerIndex) = obj.signalNum;
			
			if obj.enableTracking
				obj.trackInput{winnerIndex} = [obj.trackInput{winnerIndex}; signal];
				obj.trackInputIdx{winnerIndex} = [obj.trackInputIdx{winnerIndex}; obj.signalNum];
			end
        end

        function updateAdjacentNodes(obj, winnerIndex, signal)
            pals = find(obj.adjacencyMat(:,winnerIndex) > 0);
            for i = 1:length(pals)
                w = obj.nodes(pals(i),:);
                obj.nodes(pals(i), :) = w + (signal - w)./(100 * obj.winningTimes(pals(i)));
            end
        end

        function incrementEdgeAges(obj, winnerIndex)
            indexes = find(obj.adjacencyMat(:,winnerIndex) > 0);
            for i = 1: length(indexes)
                obj.incrementEdgeAge(winnerIndex, indexes(i));
            end
        end

        function incrementEdgeAge(obj, i, j)
            obj.adjacencyMat(i, j) = obj.adjacencyMat(i, j) + 1;
            obj.adjacencyMat(j, i) = obj.adjacencyMat(j, i) + 1;
        end

        function setEdgeAge(obj, i, j, value)
            obj.adjacencyMat(i, j) = value;
            obj.adjacencyMat(j, i) = value;
        end
        
        function winnerIndex = deleteOldEdges_Original(obj, winnerIndex)
            indexes = find(obj.adjacencyMat(:,winnerIndex) > obj.maxEdgeAge + 1); % 1 expresses that there is an edge.
            deletedNodeIndexes = [];
            for i = 1: length(indexes)
                obj.edgeDeleted = obj.edgeDeleted+1;
                obj.setEdgeAge(indexes(i), winnerIndex, 0);
                obj.setBond(indexes(i), winnerIndex, 0);
                if ~any(obj.adjacencyMat(:,indexes(i)))
                    deletedNodeIndexes = cat(1, deletedNodeIndexes, indexes(i));
                end
            end
            winnerIndex = winnerIndex - sum(deletedNodeIndexes < winnerIndex);
            obj.deleteNodes(deletedNodeIndexes);
        end

        function edgeAge = collectClusterEdgeAge(obj, seed)
            G = graph(obj.adjacencyMat);
          
            T = dfsearch(G, seed,'allevents');

            eList = unique(T.EdgeIndex);

            eList(isnan(eList)) = [];
            edgeAge = G.Edges.Weight(eList);

        end

        function winnerIndex = deleteOldEdges_Plus(obj, winnerIndex)
            % Collect all edges in cluster
            edgeAge = obj.collectClusterEdgeAge(winnerIndex);

            edge = obj.adjacencyMat(:,winnerIndex);
 
            c = prctile(edgeAge,75);
            th = obj.paramEdge*iqr(edgeAge);
            
            if th == 0
                return;
            end
            
            curTh = c + th;
            ratio = obj.edgeDeleted / ( obj.edgeDeleted + length(edgeAge));

            % Check if there are any edges to be deleted
            delThreshold = (obj.edgeAvgLtDel*ratio + curTh*(1-ratio) );
            delFlag = edge > delThreshold;
            indexes = find(delFlag);
            obj.curNodeTh = delThreshold;
            
            % Update average lifetime of deleted edges
            if ~isempty(indexes)
                obj.edgeAvgLtDel = (obj.edgeDeleted*obj.edgeAvgLtDel + sum(edge(delFlag))) / (obj.edgeDeleted + length(indexes));
            end
            
            obj.edgeDeleted = obj.edgeDeleted + length(indexes);
            deletedNodeIndexes = [];
            for i = 1: length(indexes)
                
                obj.setEdgeAge(indexes(i), winnerIndex, 0);
                if ~any(obj.adjacencyMat(:,indexes(i)))
                    deletedNodeIndexes = cat(1, deletedNodeIndexes, indexes(i));
                end
            end
            
            % Update winner index according to the deleted node
            winnerIndex = winnerIndex - sum(deletedNodeIndexes < winnerIndex);
            obj.deleteNodes(deletedNodeIndexes);
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

			if obj.enableTracking
				obj.trackInput(indexes) = [];
				obj.trackInputIdx(indexes) = [];
			end

            obj.nodeDeleted = obj.nodeDeleted + sum(indexes);

        end

        function deleteNoiseNodes_Original(obj, varargin)
            noises = sum(obj.adjacencyMat > 0) < obj.minDegree;
            
            obj.deleteNodes(noises);
        end
        
        function deleteNoiseNodes_Plus(obj, varargin)
            noises = sum(obj.adjacencyMat > 0) < obj.minDegree; 
            data = ~full(noises);
 
            IT = obj.signalNum - obj.winTS;
            UT = IT./obj.winningTimes; % unutility

            [TF, l,u,c] = isoutlier(UT(data));
            th = obj.paramAlpha*((u - c)/3);
            curTh = c + th;

            % Check if there are any nodes should be deleted
            ratio = obj.nodeDeleted / ( obj.nodeDeleted + sum(data));
            noiseLv = sum(noises) / size(obj.nodes, 1);
            delThreshold = (obj.nodeDelTh*(ratio) + curTh*(1-ratio)*(1-noiseLv) );
            inactiveIdx = UT > delThreshold;
            obj.curNodeTh = delThreshold; % tracking the main deleted threshold

            if any(inactiveIdx & noises)
                % Tracking the average deleted idle time and unutility 
                obj.nodeAvgIdleDel = (obj.nodeDeleted*obj.nodeAvgIdleDel + sum(IT(inactiveIdx & noises))) / (obj.nodeDeleted + sum(inactiveIdx & noises));
                obj.nodeDelTh = (obj.nodeDeleted*obj.nodeDelTh + sum(UT(inactiveIdx & noises))) / (obj.nodeDeleted + sum(inactiveIdx & noises));
            end
            obj.deleteNodes(inactiveIdx & noises);
        end
    end
end
