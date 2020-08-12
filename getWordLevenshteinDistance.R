#methods for extracting the edit table

getEditTable = function(s,r){
	#s and r are strings with words separated by ' '
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	a.s = strsplit(alphabet, '')[[1]]
	ss = strsplit(s,' ')[[1]]
	rr = strsplit(r,' ')[[1]]
	#if(length(ss) != length(rr)){
	#		stop('input and output must have the same number of words')
	#}	
	
	s.sym = sToSymbols(ss, a.s)	
	r.sym = rToSymbols(rr, s.sym[[3]], a.s)
	
	#weight substitutions more highly to find moves
	distObj = adist(s.sym[[1]], r.sym[[1]], counts=T,c(insertions=1, deletions=1, substitutions=2))

	editList = strsplit(attributes(distObj)$trafos[1,1],'')[[1]]
	sChar = strsplit(s.sym[[1]],'')[[1]]
	rChar = strsplit(r.sym[[1]],'')[[1]]
	
	sCounter = 1
	rCounter = 1
	rv = list()
	for(editIndex in 1:length(editList)){
		code = editList[editIndex]
		if (code %in% c('M','S') ){
			
			rv[[editIndex]] = data.frame(code = code, sWord= sChar[sCounter], rWord= rChar[rCounter],   sCounter = sCounter, rCounter = rCounter,
			sContextL1 = getContext(sChar, sCounter,-1), #1 left input	
			sContextL2 = getContext(sChar, sCounter,-2), #2 left input	
			sContextR1 = getContext(sChar, sCounter,1), #1 right input	
			sContextR2 = getContext(sChar, sCounter,2), #2 right input	
						
			rContextL1 = getContext(rChar, rCounter,-1), #1 left input	
			rContextL2 = getContext(rChar, rCounter,-2), #2 left input	
			rContextR1 = getContext(rChar, rCounter,1), #1 right input	
			rContextR2 = getContext(rChar, rCounter,2),
      sentence = s,
      response = r) #2 right input				
			sCounter =	sCounter + 1		
			rCounter =	rCounter + 1
		} else if (code == 'D'){
			rv[[editIndex]] = data.frame(code = code, sWord= sChar[sCounter], rWord= NA,   sCounter = sCounter, rCounter = NA, 
			sContextL1 = getContext(sChar, sCounter,-1), #1 left input	
			sContextL2 = getContext(sChar, sCounter,-2), #2 left input	
			sContextR1 = getContext(sChar, sCounter,1), #1 right input	
			sContextR2 = getContext(sChar, sCounter,2), #2 right input	
						
			rContextL1 = NA, #1 left input	
			rContextL2 = NA, #2 left input	
			rContextR1 = NA, #1 right input	
			rContextR2 = NA,
			sentence = s,
			response = r) #2 right input	
			sCounter= sCounter + 1		
		} else if (code == 'I'){
			rv[[editIndex]] = data.frame(code = code, sWord= NA, rWord= rChar[rCounter],   sCounter = NA, rCounter = rCounter,
			sContextL1 = NA, #1 left input	
			sContextL2 = NA, #2 left input	
			sContextR1 = NA, #1 right input	
			sContextR2 = NA, #2 right input	
						
			rContextL1 = getContext(rChar, rCounter,-1), #1 left input	
			rContextL2 = getContext(rChar, rCounter,-2), #2 left input	
			rContextR1 = getContext(rChar, rCounter,1), #1 right input	
			rContextR2 = getContext(rChar, rCounter,2),
			sentence = s,
			response = r) #2 right input	
			rCounter= rCounter + 1		
		}	
	}
	editTable = do.call('rbind', rv)

	#just give a list of columns for use with the cypher
	wordColumnNames = c('sWord','rWord')
	contextColumnNames =c('sContextL1','sContextL2', 'sContextR1','sContextR2','rContextL1','rContextL2', 'rContextR1','rContextR2') 
	columnNames	= c(wordColumnNames, contextColumnNames)
	
	for(columnName in columnNames){
		editTable[[columnName]] = symbolsToWords(editTable[[columnName]],r.sym[[4]])	
	}
	editTable$sLeftSequence = 
	unname(sapply(gsub(' *NA *','',paste(editTable$sContextL2,editTable$sContextL1, editTable$sWord)), function(x){ifelse(length(strsplit(x,' ')[[1]]) == 3, x, '')}))		
	editTable$rLeftSequence = 
	unname(sapply(gsub(' *NA *','',paste(editTable$rContextL2,editTable$rContextL1, editTable$rWord)), function(x){ifelse(length(strsplit(x,' ')[[1]]) == 3, x, '')}))
	editTable$sRightSequence = 
	unname(sapply(gsub(' *NA *','',paste(editTable$sWord,editTable$sContextR1,editTable$sContextR2)), function(x){ifelse(length(strsplit(x,' ')[[1]]) == 3, x, '')}))		
	editTable$rRightSequence = 
	unname(sapply(gsub(' *NA *','',paste(editTable$rWord, editTable$rContextR1,editTable$rContextR2)), function(x){ifelse(length(strsplit(x,' ')[[1]]) == 3, x, '')}))
	
	for (cn in contextColumnNames){editTable[[cn]] = NULL}
	
	editTable$code = as.character(editTable$code)
	return(editTable)	
}

sToSymbols = function(ss, a.s){
#use factors
	waf = factor(ss)
	returnString = paste(a.s[as.numeric(waf)], collapse='')
	cypher = data.frame(letter=a.s[as.numeric(waf)], word=as.character(levels(waf)[as.numeric(waf)]), stringsAsFactors=F)	
return(list(returnString, cypher, levels(waf)))	
}


rToSymbols = function(rr, sLevels, a.s){
	alreadyNamed = as.numeric(factor(rr, levels=sLevels))
	requiringNames = rr[which(is.na(alreadyNamed))]
	newFactors = factor(requiringNames)
	newIndices = as.numeric(newFactors) + length(sLevels)
	newLevels = c(sLevels, levels(newFactors))
	
	waf = factor(rr, newLevels)
	returnString = paste(a.s[as.numeric(waf)], collapse='')
	cypher = data.frame(letter=a.s[as.numeric(waf)], word=as.character(levels(waf)[as.numeric(waf)]), stringsAsFactors=F)
	completeCypher = data.frame(letter=a.s[1:length(newLevels)], word=as.character(newLevels), stringsAsFactors=F)
	
	return(list(returnString, cypher, levels(waf), completeCypher))	
}

reduceBracketedChanges = function(editTable){
	editString = paste(editTable$code, collapse='')
	
	raw_DI_start_indices = (gregexpr('(^|M)DI($|M)',editString))[[1]]	
	
	DI_start_indices = sapply(raw_DI_start_indices, function(x){
		if (editTable$code[x] == 'M'){
			return(x+1)
		} else {
			return(x)
		}		
	})

	output = list()
	output_index = 0 
	input_index = 0

	while (input_index < nrow(editTable)){
		input_index = input_index + 1
		output_index = output_index + 1
		if (input_index %in% DI_start_indices){
			# if an index is in start_indices, grab two, reduce, and add to the output
			output[[output_index]] = reduceDI(editTable[input_index:(input_index+1),])			
			# increment one more in the input
			input_index = input_index + 1
		} else {
			# if it is not, grab one and add it to the output			
			output[[output_index]] =	 editTable[input_index,]		
		}
	}
	return(do.call("rbind", output))	
}	
reduceDI = function(et){
	return(data.frame(code='S', sWord=et$sWord[1], rWord=et$rWord[2], sCounter=et$sCounter[1], rCounter=et$rCounter[2], sentence= et$sentence[1], response=et$response[1],sLeftSequence = et$sLeftSequence[1], rLeftSequence = et$rLeftSequence[2], sRightSequence = et$sRightSequence[1], rRightSequence = et$rRightSequence[2],stringsAsFactors=F))
}	
	

getReducedEditTable = function(s,r){
	initial_edit_table = getEditTable(s,r)
	reduceBracketedChanges(initial_edit_table)
}

getContext = function(charArray,counter,position){
	#position is -2,-1,1, 2	
	gram = charArray[ifelse(max(counter+ position,0) <= 10,max(counter + position,0),0)]	
	if (length(gram) > 0 ){
		return(gram)
	} else {
		return(NA)
	}
}

symbolsToWords = function(toTranslate, cypher){
	t1 = sapply(toTranslate, function(x){subset(cypher, as.character(letter) == as.character(x))$word})
	t1[!as.logical(sapply(t1, function(x){length(nchar(x))}))] = NA
	return(unlist(t1))
}
