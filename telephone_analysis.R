library('lme4')
library('lmerTest')
library('Hmisc')

ensureTrailingSlash= function(string){
	#make sure that there is a trailing slash in a filename
	if (substr(string, nchar(string),nchar(string)) != '/'){
		return(paste(string,'/', sep=''))
	} else {
		return(string)
	}	
}

RtexVars = function(keys, values, outfile){
    if(length(keys) != length(values)){
        stop('keys and values must be of the same length')
    }
    cat(paste(paste('\\newcommand{\\',keys, '}{', values, '}', sep=''), collapse ='\n'), file = outfile)	
}

getSigStatementTable = function(x){
    return(
        ifelse(x > .05, paste(round(x,digits=1),sep=""), 
                     ifelse(x < .0001, "\\textbf{<.0001}", 
                            ifelse(x < .001,"\\textbf{<.001}",
                                   ifelse(x < .01, "\\textbf{<.01}", "\\textbf{<.05}"))))
    )
}

getSigStatementInline = function(x){
    return(
        ifelse(x > .05, paste("=", round(x,digits=1)), 
                     ifelse(x < .0001, "$<$ .0001", 
                            ifelse(x < .001,"$<$ .001",
                                   ifelse(x < .01, "$<$ .01", "$<$ .05"))))
    )
}


modelToTable <- function(type, modelName, lm1.lmerTest, lm2.lmer=NULL, replacements=NULL, file="", printVars = F,...) {
    # type is the kind of model
    # lm1.lmerTest is the model, augmented with statistical tests from lmerTest
    # lm2 is a standard lme4 model (easier to extract vars)
    # replacments is a named list with form {searchTerm: replacement}
    # file is the output file for the LM table
    # ... passes any vars along to Hmisc for output to latex  

    coefs = as.data.frame(coef(summary(lm1.lmerTest)))

    # update the predictor names coming through the replacments dict
    prednames <- row.names(coefs)
    if (!is.null(replacements)){
        for (searchTerm in names(replacements)){
            #print('Searching for:')
            #print(searchTerm)
            #print('replacing with:')
            #print(replacements[[searchTerm]])
            prednames = gsub(searchTerm,replacements[[searchTerm]], prednames)
        }    
    }
    print(prednames)
    
    coefs$df = NULL
    coefs[,1] = round(coefs[,1],digits=4)
    coefs[,2] = round(coefs[,2],digits=4)
    coefs[,3] = round(coefs[,3],digits=2)  
    sigs = coefs[,4]
    coefs[,4] = round(sigs,digits=2)
    coefs[,4] = sapply(sigs, getSigStatementTable)    
    row.names(coefs) = prednames

    if (type == 'fixed_linear'){
                                     
        colnames(coefs) = c("Coef $\\beta$","SE($\\beta$)","\\textit{t} value","$p$")
        rdf = coefs
        
        keys  = paste0(modelName,gsub('[[:punct:]]| |[0-9]','',prednames))
        inlineSigs = sapply(sigs, getSigStatementInline)
        vals = paste0('(',"$\\beta$ = ",coefs[,1], ", \\textit{t} value = ", coefs[,3],", $p$ ", inlineSigs,')')  
        

    } else if (type == 'mixed_linear'){
        #random effects doesn't handle slopes
        colnames(coefs) = c("Coef $\\beta$","SE($\\beta$)","\\textit{t} value","$p$")
        
        print(attributes(VarCorr(lm1.lmerTest)))

        mixed_effects_names = attributes(VarCorr(lm1.lmerTest))$names
        #these are the names of the random factors

        random_vars = data.frame()
        for (mixed_effect_name in mixed_effects_names){ 
            # determining factor is the number of standard deviation entries
            std_dev = round(attributes(VarCorr(lm1.lmerTest)[[mixed_effect_name]])$stddev, 2)
            var = names(attributes(VarCorr(lm1.lmerTest)[[mixed_effect_name]])$stddev)          
            
            # augment with the variable names
            rdf = data.frame(std_dev, var)
            rdf$factorName = mixed_effect_name            
            random_vars= rbind.fill(random_vars, rdf[,c('factorName', 'var', 'std_dev')])
        }
        
        
        # replace any factor names with readable replacements
        if (!is.null(replacements)){
            for (searchTerm in names(replacements)){
                #print('Searching for:')
                #print(searchTerm)
                #print('replacing with:')
                #print(replacements[[searchTerm]])
                random_vars$factorName = gsub(searchTerm,replacements[[searchTerm]], random_vars$factorName)
                random_vars$var = gsub(searchTerm,replacements[[searchTerm]], random_vars$var)
            }    
        }

        print(random_vars)
        
        # move the titles into the columns
        random_vars$rownames = paste(random_vars$var,'$|$', random_vars$factorName)
        rownames(random_vars) = random_vars$rownames
        random_vars$factorName = NULL
        random_vars$var = NULL
        random_vars$rownames = NULL
        
        # pad to the width of coefs
        padIndex = 0 
        while (ncol(random_vars) < ncol(coefs)){
            padIndex = padIndex+1
            random_vars[[paste0('pad', padIndex)]] = ''
        }
        
        # rename columns
        names(random_vars) = names(coefs)    
    
        #make a line to describe random effects
        randomEffectsLabels = data.frame(mat.or.vec(2, ncol(coefs)))   
        randomEffectsLabels[1:nrow(randomEffectsLabels),1:ncol(randomEffectsLabels)] = ''     
        names(randomEffectsLabels) = names(coefs)    
        randomEffectsLabels[2,1] = 'Std. Dev'
        rownames(randomEffectsLabels) = c('Random Effects','')

        rdf = rbind(coefs, randomEffectsLabels, random_vars)
        print(rdf)

        fixed_keys  = paste0(modelName,gsub('[[:punct:]]| |[0-9]','',prednames))
        inlineSigs = sapply(sigs, getSigStatementInline)
        fixed_vals = paste0('(',"$\\beta$ = ",coefs[,1], ", \\textit{t} value = ", coefs[,3],", $p$ ", inlineSigs,')')  

        random_keys = paste0(modelName,gsub('[[:punct:]]| |[0-9]','',rownames(random_vars)))
        random_vals = paste0('(Std. Dev. = ',random_vars[,1])

        keys = c(fixed_keys, random_keys)
        vals = c(fixed_vals, random_vals)


    } else if (type == 'fixed_logistic'){
        stop('Not Implmented')
    
    } else if (type == 'mixed_logistic'){
        colnames(coefs) = c("Coef $\\beta$","SE($\\beta$)","\\textit{z} value","$Pr(>|\\textit{z}|)$")

        print(attributes(VarCorr(lm1.lmerTest)))

        mixed_effects_names = attributes(VarCorr(lm1.lmerTest))$names
        
        mixed_effects_aliases = mixed_effects_names
        if (!is.null(replacements)){
            for (searchTerm in names(replacements)){
                #print('Searching for:')
                #print(searchTerm)
                #print('replacing with:')
                #print(replacements[[searchTerm]])
                mixed_effects_aliases = gsub(searchTerm,replacements[[searchTerm]], mixed_effects_aliases)
            }    
        }
        print('Alisases for random variables')
        print(mixed_effects_aliases)
        
        mixed_effects_std_devs = c()
        for (mixed_effect_name in mixed_effects_names){
            mixed_effects_std_devs = c(mixed_effects_std_devs, round(attributes(VarCorr(lm1.lmerTest)[[mixed_effect_name]])$stddev, 2))
        }

        #!!! automate this step of extraction
        randCoefs = data.frame(      
        c('Std. Dev.', mixed_effects_std_devs),
        # pad so that it is equally wide as the fixed effects
        null1 = c('','',''), 
        null2 = c('','',''),
        null3 = c('','',''))
        names(randCoefs) = colnames(coefs) #not the real labels, but necessary to get into the same table with Hmisc
        rownames(randCoefs) = c('',mixed_effects_aliases) # first one is the Std. deviation title
                 
        rdf = rbind(coefs, randCoefs)

        fixed_keys  = paste0(modelName,gsub('[[:punct:]]| |[0-9]','',prednames))
        inlineSigs = sapply(sigs, getSigStatementInline)
        fixed_vals = paste0('(',"$\\beta$ = ",coefs[,1], ", \\textit{z} value = ", coefs[,3],", $p$", inlineSigs,')')  

        random_keys = paste0(modelName,gsub('[[:punct:]]| |[0-9]','',mixed_effects_aliases))
        random_vals = paste0('(Std. Dev. = ',mixed_effects_std_devs)

        keys = c(fixed_keys, random_keys)
        vals = c(fixed_vals, random_vals)

    }
    
    latex(rdf,file=file,title="",table.env=FALSE,booktabs=TRUE, ...)
    if (printVars){
        RtexVars(keys, vals, gsub('_lm','_vars',file))
    }

}