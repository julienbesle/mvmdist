% Generates a random sample from a mixture of one circular uniform distribution and one or several equidistant von Mises distribution(s)
% and fits mixture models including from 1 to maxNpeaks vonMises distributions, with or without including a circular uniform distribution.
% Plots the density histogram and functions of the sample and best-fitting models (aveaged across a number of repeats
% and calculate their Bayesian and corrected Akaike information criteria, as well as the respective posterior probabilities
% of each model with respect to that of with one less mixture component.
%
% author: julien besle (07/08/2020)

% parameters
nSamples = 1000; % total number of samples per experiment
nRepeats = 1; %number of random samples drawn from the mixture (number of experiments)
proportions = [.3 .3]; % proportion of samples from each von Mises distribution. Any shortfall from 1 is sampled from the random uniform distribution
maxNpeaks = 6; % total number of fitted distributions (including the uniform one)
nSubploLines = 2; %number of rows of plots in figure

if isempty(which('num2words'))
  num2words = @mat2str;
end

% generate mixture of distributions
nVMs = length(proportions); % number of von Mises distributions in the mixture model
vmMus = linspace(-pi+pi/nVMs,pi-pi/nVMs,nVMs); % equidistant centres of the distributions
vmModel = VonMisesMixture([1-sum(proportions) proportions],[0 vmMus],[0 ones(1,nVMs)]);% create von Mises mixture model

nBins = min(40,round(nSamples/20)); % number of bins for histogram
angles = (-pi:pi/100:pi)'; % angle values for plotting model pdfs
nAngles = size(angles,1);
seed = rng; % get random generator seed
for includeCircularUniform = [false true]
  
  if includeCircularUniform
    uniformString = '';
  else
    uniformString = 'NOT ';
  end
  figure('name',sprintf('One circular uniform (%s%%) and %s von Mises distributions centered at %s, proportions = %s%%, %d samples | Uniform distribution %smodelled', ...
                mat2str((1-sum(proportions))*100,3),num2words(nVMs),mat2str(vmMus,3),mat2str(proportions*100,2),nSamples,uniformString), ...
                'unit','normalized','position',[0 0 1 1]);

  rng(seed) %reset random generator
  % initialize variables
  sampleHist = nan(nBins,nRepeats);
  for iPeak = 1:maxNpeaks
    mu{iPeak} = zeros(nRepeats,iPeak);
    kappa{iPeak} = zeros(nRepeats,iPeak);
    componentProportion{iPeak} = zeros(nRepeats,iPeak);
    logLikelihood{iPeak} = zeros(nRepeats,iPeak);
    vmmPdf{iPeak} = zeros(nAngles,nRepeats);
  end
  BIC = inf(nRepeats,maxNpeaks);
  AICC = inf(nRepeats,maxNpeaks);
  bayesFactorBIC = ones(1,maxNpeaks);
  bayesFactorAICC = ones(1,maxNpeaks);
  posteriorBIC = 0.5*ones(1,maxNpeaks);
  posteriorAICC = 0.5*ones(1,maxNpeaks);
  
  % draw sample and fit mixtures
  for iRepeat = 1:nRepeats
    
    sample = vmModel.random(nSamples,false); % draw random values from the mixture (including the uniform distribution)
    [sampleHist(:,iRepeat),binEdges] = histcounts(sample,nBins,'binLimits',[-pi pi], 'Normalization', 'pdf'); % create sample density histogram for averaging across repreats

    thisMmFit = cell(maxNpeaks,1);
    for iPeak = 1:maxNpeaks
      % fit mixture model to sample
      thisMmFit{iPeak} = fitmvmdist(sample, iPeak, 'IncludeCircularUniform', includeCircularUniform);
      
      % keep fitted parameters for averaging
      mu{iPeak}(iRepeat,:) = thisMmFit{iPeak}.mu';
      kappa{iPeak}(iRepeat,:) = thisMmFit{iPeak}.kappa';
      componentProportion{iPeak}(iRepeat,:) = thisMmFit{iPeak}.componentProportion';
      
      % update Bayes factors
      nParameters = iPeak*3 - 2*includeCircularUniform - 1; % the uniform distribution has only one parameter (its contribution to the mixture), instead of 3 for a von Mises
      BIC(iRepeat,iPeak) = nParameters * log(nSamples) - 2*thisMmFit{iPeak}.logLikelihood;
      AICC(iRepeat,iPeak) = 2 * nParameters - 2 * thisMmFit{iPeak}.logLikelihood + 2*nParameters*(nParameters+1)/(nSamples-nParameters-1);

      if iPeak>1
        bayesFactorBIC(iPeak) = bayesFactorBIC(iPeak) * exp((BIC(iRepeat,iPeak-1) - BIC(iRepeat,iPeak))/2); % accumulate evidence across repeats
        bayesFactorAICC(iPeak) = bayesFactorAICC(iPeak) * exp((AICC(iRepeat,iPeak-1) - AICC(iRepeat,iPeak))/2);
      end
      vmmPdf{iPeak}(:,iRepeat) = pdf(thisMmFit{iPeak}, angles); % save fitted model's PDF
    end
  end
  
  % average BIC and AICc (there might be a more correct way to combine them)
  BIC = mean(BIC,1); %
  AICC = mean(AICC,1); %
  
  % average histogram across repeats
  sampleHist = mean(sampleHist,2);
  
  for iPeak = 1:maxNpeaks

    % average model PDFs across repeats
    averageVmmPdf = mean(vmmPdf{iPeak},2);
    
    % create average mixture model
    [mu{iPeak}(:,1+includeCircularUniform:end),sortingIndices] = sort(mu{iPeak}(:,1+includeCircularUniform:end),2); %sort components by increasing mu
    for iRepeat = 1:nRepeats %apply sorting to other parameters
      kappa{iPeak}(iRepeat,1+includeCircularUniform:end) = kappa{iPeak}(iRepeat,includeCircularUniform+sortingIndices(iRepeat,:));
      componentProportion{iPeak}(iRepeat,1+includeCircularUniform:end) = componentProportion{iPeak}(iRepeat,includeCircularUniform+sortingIndices(iRepeat,:));
    end
    mmFit{iPeak} = VonMisesMixture(mean(componentProportion{iPeak},1), angle(mean(exp(1i*mu{iPeak}),1)), mean(kappa{iPeak},1)); % create average 
    averageParamsVmmPdf = pdf(mmFit{iPeak}, angles);
    
    % compute posterior probabilities
    posteriorBIC(iPeak) = 1./(1+1/bayesFactorBIC(iPeak));
    posteriorAICC(iPeak) = 1./(1+1/bayesFactorAICC(iPeak));
    
    % plot average histograms and models
    subplot(nSubploLines,ceil(maxNpeaks/nSubploLines),iPeak);
    histogram('BinEdges',binEdges,'binCounts',sampleHist,'edgecolor','none','facecolor',[.6 .6 .6]); %plot average density histogram
    hold('on');
    plot(angles,averageVmmPdf,'linewidth',2); % plot average model PDF
    plot(angles,averageParamsVmmPdf,'linewidth',2); % plot PDF of average model (model with average parameters)
    % plot centers of von Mises distribution (from average model) as vertical bars and add average kappa and proportion parameters 
    for jPeak = 1+includeCircularUniform:iPeak
      [~,whichAngle] = min(abs(angles- mmFit{iPeak}.mu(jPeak)));
      maxY = averageParamsVmmPdf(whichAngle);
      plot(mmFit{iPeak}.mu(jPeak)*ones(2,1),[0;maxY],'linewidth',2);
      text(mmFit{iPeak}.mu(jPeak),maxY/2,sprintf('k=%.2f',mmFit{iPeak}.kappa(jPeak)),'HorizontalAlignment','center');
      text(mmFit{iPeak}.mu(jPeak),maxY/2-.015,sprintf('(%s%%)',mat2str(mmFit{iPeak}.componentProportion(jPeak)*100,3)),'HorizontalAlignment','center');
    end
    if includeCircularUniform % add kappa (0) and proportion parameters of circular uniform distribution
      text(pi,mean(pdf(mmFit{iPeak}, angles)),sprintf('k=%.0f',mmFit{iPeak}.kappa(1)),'HorizontalAlignment','left');
      text(pi,mean(pdf(mmFit{iPeak}, angles))-.015,sprintf('(%s%%)',mat2str(mmFit{iPeak}.componentProportion(1)*100,3)),'HorizontalAlignment','left');
    end
    yLim = get(gca,'ylim');
    yLim(2) = 1.3*yLim(2); %add some apace at the top of the plot
    set(gca,'ylim',yLim);
    % display BIC/AICC and posterior probabilities
    sortedBIC = sort(BIC);
    sortedAICC = sort(AICC);
    text(1,.9*yLim(2),sprintf('BIC = %.0f',BIC(iPeak)),'HorizontalAlignment','right','color',[BIC(iPeak)==sortedBIC(1) (BIC(iPeak)==sortedBIC(2))*.8 0]); % smallest BIC in red and second smallest in green
    text(3,.9*yLim(2),sprintf('AICc = %.0f',AICC(iPeak)),'HorizontalAlignment','right','color',[AICC(iPeak)==sortedAICC(1) (AICC(iPeak)==sortedAICC(2))*.8 0]);
    if iPeak>1
      text(1,.8*yLim(2),sprintf('Post. Prob. (w/r n-1): %.2f',posteriorBIC(iPeak)),'HorizontalAlignment','right');
      text(3,.8*yLim(2),sprintf('%.2f',posteriorAICC(iPeak)),'HorizontalAlignment','right');
    end
    set(gca,'xlim',[-pi pi]);
  end
end
