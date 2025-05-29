# data
amazonas <- c(497,1,2,4,1,4,96,10,5,6,23,1,1,3,1,3,7,1,35,6,2,2,9,2,30,3)
puno <- c(4,1,1,5,1,1,1,1,13,168,1,1,23,2,38,2,4,1,1,17,1,63,58,1,11,21,8,1324,5)

# mean
m_amaz<- mean(amazonas)
m_pun<- mean(puno)

#desv
desv_amaz<- sd(amazonas)
desv_pun<- sd(puno)

# test t de Welch (unequal variances)
t_test <- t.test(amazonas, puno, var.equal = FALSE)

# results
print(t_test)
