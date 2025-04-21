library(shiny)
library(readxl)
library(ggplot2)
library(report)

ui <- fluidPage(
  titlePanel("Pruebas de Hipótesis desde Excel"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("archivo", "Cargar archivo Excel", accept = c(".xls", ".xlsx")),
      uiOutput("selector_columnas"),
      selectInput("operacion", "Prueba de hipótesis",
                  choices = list(
                    "Medias" = list(
                      "Z para una media" = "z_una_media",
                      "t para una media" = "t_una_media",
                      "t para dos medias independientes" = "t_independiente",
                      "t para dos medias pareadas" = "t_pareada"
                    ),
                    "Proporciones" = list(
                      "Z para una proporción" = "z_una_prop",
                      "Z para dos proporciones" = "z_dos_prop"
                    ),
                    "Varianzas" = list(
                      "Chi-cuadrado para una varianza" = "chi_varianza",
                      "F para comparar dos varianzas" = "f_varianzas"
                    ),
                    "Normalidad" = list(
                      "Shapiro-Wilk" = "shapiro",
                      "Kolmogorov-Smirnov" = "ks"
                    ),
                    "ANOVA" = list(
                      "ANOVA de un factor" = "anova"
                    )
                  )),
      actionButton("calcular", "Calcular")
    ),
    
    mainPanel(
      verbatimTextOutput("resultado"),
      plotOutput("grafico")
    )
  )
)
server <- function(input, output, session) {
  datos <- reactive({
    req(input$archivo)
    read_excel(input$archivo$datapath)
  })
  
  output$selector_columnas <- renderUI({
    req(datos())
    selectInput("columnas", "Selecciona columnas", choices = names(datos()), multiple = TRUE)
  })
  
  observeEvent(input$calcular, {
    req(input$columnas)
    df <- datos()
    cols <- input$columnas
    oper <- input$operacion
    
    output$grafico <- renderPlot(NULL)
    output$resultado <- renderPrint(NULL)
    
    resultado <- NULL
    
    # Comprobaciones comunes
    validar_numerica <- function(x) {
      is.numeric(df[[x]])
    }
    
    validar_factor <- function(x) {
      is.factor(df[[x]]) || is.character(df[[x]])
    }
    
    switch(oper,
           
           # --- MEDIAS ---
           "z_una_media" = {
             if (length(cols) != 1 || !validar_numerica(cols[1])) {
               resultado <- "❌ Z para una media requiere UNA columna numérica."
             } else {
               x <- df[[cols[1]]]
               resultado <- t.test(x, mu = mean(x), var.equal = TRUE)
             }
           },
           
           "t_una_media" = {
             if (length(cols) != 1 || !validar_numerica(cols[1])) {
               resultado <- "❌ t para una media requiere UNA columna numérica."
             } else {
               x <- df[[cols[1]]]
               resultado <- t.test(x, mu = mean(x))
             }
           },
           
           "t_independiente" = {
             if (length(cols) != 2 || !validar_numerica(cols[2]) || !validar_factor(cols[1])) {
               resultado <- "❌ t para dos medias independientes requiere una columna categórica (grupo) y otra numérica."
             } else {
               grupo <- as.factor(df[[cols[1]]])
               x <- df[[cols[2]]]
               resultado <- t.test(x ~ grupo)
             }
           },
           
           "t_pareada" = {
             if (length(cols) != 2 || !all(sapply(cols, validar_numerica))) {
               resultado <- "❌ t pareada requiere DOS columnas numéricas relacionadas."
             } else {
               x1 <- df[[cols[1]]]
               x2 <- df[[cols[2]]]
               resultado <- t.test(x1, x2, paired = TRUE)
             }
           },
           
           # --- PROPORCIONES ---
           "z_una_prop" = {
             if (length(cols) != 1 || !all(df[[cols[1]]] %in% c(0, 1))) {
               resultado <- "❌ Z para una proporción requiere una columna con datos binarios (0 y 1)."
             } else {
               prop <- sum(df[[cols[1]]])
               total <- length(df[[cols[1]]])
               resultado <- prop.test(prop, total, p = 0.5)
             }
           },
           
           "z_dos_prop" = {
             if (length(cols) != 2 || !all(sapply(cols, function(x) length(unique(df[[x]])) == 2))) {
               resultado <- "❌ Z para dos proporciones requiere dos columnas categóricas binarias."
             } else {
               tabla <- table(df[[cols[1]]], df[[cols[2]]])
               resultado <- prop.test(tabla)
             }
           },
           
           # --- VARIANZAS ---
           "chi_varianza" = {
             if (length(cols) != 1 || !validar_numerica(cols[1])) {
               resultado <- "❌ Chi-cuadrado requiere UNA columna numérica."
             } else {
               x <- df[[cols[1]]]
               n <- length(x)
               var_obs <- var(x)
               sigma0 <- 1  # valor de comparación teórico
               chi_val <- (n - 1) * var_obs / sigma0^2
               p_valor <- 1 - pchisq(chi_val, df = n - 1)
               resultado <- list("Estadístico Chi-cuadrado" = chi_val,
                                 "Grados de libertad" = n - 1,
                                 "Valor p" = p_valor)
             }
           },
           
           "f_varianzas" = {
             if (length(cols) != 2 || !all(sapply(cols, validar_numerica))) {
               resultado <- "❌ Prueba F requiere dos columnas numéricas para comparar varianzas."
             } else {
               x1 <- df[[cols[1]]]
               x2 <- df[[cols[2]]]
               resultado <- var.test(x1, x2)
             }
           },
           
           # --- NORMALIDAD ---
           "shapiro" = {
             if (length(cols) != 1 || !validar_numerica(cols[1])) {
               resultado <- "❌ Shapiro-Wilk requiere una columna numérica."
             } else {
               x <- df[[cols[1]]]
               resultado <- shapiro.test(x)
             }
           },
           
           "ks" = {
             if (length(cols) != 1 || !validar_numerica(cols[1])) {
               resultado <- "❌ Kolmogorov-Smirnov requiere una columna numérica."
             } else {
               x <- df[[cols[1]]]
               resultado <- ks.test(x, "pnorm", mean(x), sd(x))
             }
           },
           
           # --- ANOVA ---
           "anova" = {
             if (length(cols) != 2 || !validar_factor(cols[1]) || !validar_numerica(cols[2])) {
               resultado <- "❌ ANOVA requiere una columna categórica (grupo) y una numérica (variable)."
             } else {
               grupo <- as.factor(df[[cols[1]]])
               y <- df[[cols[2]]]
               modelo <- aov(y ~ grupo)
               resultado <- summary(modelo)
             }
           }
    )
    
    output$resultado <- renderPrint({ resultado })
  })
}

shinyApp(ui, server)

