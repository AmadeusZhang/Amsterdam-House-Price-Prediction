
# 0. Required packages ----------------------------------------------------

library( car )
library( ggplot2 )
library( GGally )
library( reshape )
library( corrplot )
library( MASS )
library( leaps )
library( caret )


# 1. Data Importing -------------------------------------------------------

data = read.csv( file = "Source/HousingPrices-Amsterdam-August-2021.csv", header = T )
head( data )
summary( data )

# 4 NA in 'Price'

data = na.omit( data )

dim( data )

data = subset( data, select = -c( X, Address, Zip ) ) # unique values


# 2. Data Visualization ---------------------------------------------------

attach( data )

par( mfrow = c(1, 2) )
hist( Price, xlab = "House price", main = "Histogram" )
boxplot( Price, xlab = "House price", main = "Boxplot" )
dev.off()

# ggplot( data, aes( Price ) ) +
#   geom_histogram( aes( y = ..density.. ), fill = "lavender", col = 'black' ) +
#   stat_function( fun = dnorm, args = list( mean = mean( Price ), sd = sd( Price ) ), size = 0.75 ) +
#   geom_vline( xintercept = mean( Price ), color = 'red', size = 0.75 ) +
#   labs( title = 'Histogram with normal curve' ) +
#   xlab( 'House Price' )

ggpairs(
  data = data,
  title = "ReLationships between predictors & response",
  lower = list( continuos = wrap("points", alpha = 0.5, size = 0.1 ) )
)

detach( data )

# 3. Linear model ---------------------------------------------------------

g = lm( Price ~ Area + Room + Lon + Lat, data = data )
summary( g )
AIC( g )
vif( g )

par( mfrow = c(2,2) )
plot( g )
dev.off()

# Step wise
step( g ) # don't suggest us to remove predictors, but reminder it's highly influenced by outliers

# Collinearity
X = subset( data, select = -c( Price ) )
corrplot( cor( X ), method = 'color', addCoef.col = 'black' )

g = lm( Price ~ Area + Room + Lon + Lat, data = data )
summary( g )

# 'Lon' and 'Lat' area removed

g = lm( Price ~ Area + Room, data = data )
gs = summary( g )


# 4. Hypotheses --------------------------------------------------------------

attach( g )

# a. Normality:
qqnorm( residuals, ylab = "Raw Residuals", main = "Normal Q-Q Plot" )
qqline( residuals, col = "red" )
shapiro.test( residuals )
# --> hypothesis is vioLated

# b. Homoschedasticity:
plot( fitted.values, residuals,
      xlab = "Fitted Values", ylab = "Residuals",
      main = "Residuals vs Fitted Values", pch = 16 )

abline( h = 0, lwd = 2, lty = 2, col = "red" )


# 5. Data Preprocessing ---------------------------------------------------

# a. Leverage:
lev = hatvalues( g )
p = rank
n = dim( data )[1]

plot( fitted.values, lev, xlab = "Fitted Values", ylab = "Leverages", main = "Plot of leverages" )
abline( h = 2 * p/n, lty = 2, col = 'red' )

watchout_points_lev = lev[ which( lev > 2 * p/n  ) ]
watchout_ids_lev = seq_along( lev )[ which( lev > 2 * p/n ) ]

points( fitted.values[ watchout_ids_lev ], watchout_points_lev, col = 'red', pch = 16 )

pts_leverages = data[ lev > 2 * p/n, ] # [70] -> [48] (dopo introduzione di interazione) -> [47]

# Fitting the model without leverages:
g_leverages = lm( Price ~ Area + Room, data = data, subset = ( lev < 2 * p/n ) )
summary( g_leverages ) # calo di R^2, ragionevole, xk leverages tirano retta di fitting verso di sé
shapiro.test( g_leverages$residuals )

# variation of coef:
abs( (g$coefficients - g_leverages$coefficients) / g$coefficients )


  # b. Standardized Residuals:
res_std = residuals / summary(g)$sigma

watchout_ids_rstd = which( abs( res_std ) > 2 )
watchout_rstd = res_std[ watchout_ids_rstd ]

plot( fitted.values, res_std,
      xlab = "Fitted Values", ylab = "Standardized Residuals", main = "Standardized Residuals" )

abline( h = c(-2,2), lty = 2, col = 'orange' )

points( fitted.values[ watchout_ids_rstd ], 
        res_std[ watchout_ids_rstd ], col = 'red', pch = 16 )

points( fitted.values[ watchout_ids_lev ], 
        res_std[ watchout_ids_lev ], col = 'orange', pch = 16 )

legend('topright', col = c('red','orange'), 
       c('Standardized Residuals', 'Leverages'), pch = rep( 16, 2 ), bty = 'n' )

pts_stdres = data[ abs( res_std ) > 2, ] # [32] -> [32] -> [31]

# Fitting the model with res_std :
g_stdres = lm( Price ~ Area + Room + Lat, data = data[ abs(res_std) < 2, ] )
summary( g_stdres )
shapiro.test( g_stdres$residuals )


# c. Studentized Residuals:
stud = rstandard( g )

watchout_ids_stud = which( abs( stud ) > 2 )
watchout_stud = stud[ watchout_ids_stud ]

plot( fitted.values, stud, 
      xlab = "Fitted Values", ylab = "Studentized Residuals", main = "Studentized Residuals", pch = 16 )

abline( h = c(-2,2), lty = 2, col = 'orange' )

points( fitted.values[ watchout_ids_stud ], stud[ watchout_ids_stud ], col = 'blue', pch = 16 )

points( fitted.values[watchout_ids_rstd ], res_std[ watchout_ids_rstd ], col = 'red', pch = 16 )

points( fitted.values[ watchout_ids_lev ], stud[ watchout_ids_lev ], col = 'orange', pch = 16 )

legend('topright', col = c('blue','red','orange'), 
       c('Studentized Residual', 'Standardized Residuals', 'Leverages'), pch = rep( 16, 3 ), bty = 'n' )

# Studentized residuals and Standardized residuals identify the same influential points in this case.


# d. Cook's Distance:
Cdist = cooks.distance( g )

watchout_ids_Cdist = which( Cdist > 4/(n-p) ) 
watchout_Cdist = Cdist[ watchout_ids_Cdist ]

par( mfrow = c( 1, 3 ) )

plot( fitted.values, Cdist, pch = 16, 
      xlab = 'Fitted values', ylab = 'Cooks Distance', main = 'Cooks Distance' )

points( fitted.values[ watchout_ids_Cdist ], Cdist[ watchout_ids_Cdist ], col = 'green', pch = 16 )

plot( fitted.values, stud, pch = 16,
      xlab = 'Fitted values', ylab = 'Studentized Residuals', main = 'Studentized Residuals' )

points( fitted.values[ watchout_ids_stud ], stud[ watchout_ids_stud ], col = 'blue', pch = 16 )

plot( fitted.values, lev, pch = 16,
      xlab = 'Fitted values', ylab = 'Leverages', main = 'Leverages' )

points( fitted.values[ watchout_ids_lev ], lev[ watchout_ids_lev ], col = 'orange', pch = 16 )

dev.off()

pts_cdist = data[ Cdist > 4 / (n-p), ] # [46] -> [42] -> [41]

# Fitting the model with Cook's Distance:
g_cdist = lm( Price ~ Area + Room, data = data[ Cdist < 4 / (n-p), ])
summary( g_cdist )
shapiro.test( g_cdist$residuals )

# e. Linear Model with application of techniques:
g = lm( Price ~ Area + Room, data = data )
gs = summary( g )

intersect_points = Reduce( intersect, list( watchout_ids_Cdist, watchout_ids_lev, watchout_ids_stud ) )
head( data[ intersect_points, ] )

id_to_keep = !(1:n %in% intersect_points)
preprocessed_data = data[ id_to_keep, ]
g = lm( Price ~ Area + Room, data = preprocessed_data )
gs = summary(g)
shapiro.test( g$residuals )

# 6. Transformation -------------------------------------------------------

# BOX-COX: (does not work)
bc_trans = boxcox( g )
lambda_index = which.max( bc_trans$y )
lambda = bc_trans$x[ lambda_index ]

# since lambda is close to 0: (lambda = 0.06 is hard to interpreter)

# Fitting the model according to logaritmic transformation:
g_trans = lm( log(Price) ~ Area + Room, data = preprocessed_data )
summary( g_trans )
shapiro.test( g_trans$residuals )

# Confront normality:
par( mfrow = c(1,2) )
qqnorm( g$residuals )
qqline( g$residuals )
qqnorm( g_trans$residuals )
qqline( g_trans$residuals)
dev.off()
