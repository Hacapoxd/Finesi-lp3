library(shiny)
library(readxl)
library(ggplot2)
library(report)

ui <- fluidPage(
  titlePanel("Prueba T y ANOVA desde Excel"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("archivo", "Cargar archivo Excel", accept = c(".xls", ".xlsx")),
      uiOutput("selector_columnas"),
      selectInput("operacion", "Operación",
                  choices = c("Prueba T" = "t", "ANOVA" = "anova")),
      actionButton("calcular", "Calcular")
    ),
    
    mainPanel(
      verbatimTextOutput("resultado"),
      verbatimTextOutput("r"),
      verbatimTextOutput("resumen_modelo"),
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
    selectInput("columnas", "Selecciona columnas (primera = grupo)",
                choices = names(datos()), multiple = TRUE)
    
  })
  
  observeEvent(input$calcular, {
    req(input$columnas)
    df <- datos()
    cols <- input$columnas
    oper <- input$operacion
    
    if (length(cols) < 2) {
      output$resultado <- renderText("Selecciona al menos una columna de grupo y una variable numérica.")
      output$grafico <- renderPlot(NULL)
      return()
    }
    
    grupo <- as.factor(df[[cols[1]]])
    variables <- cols[-1]
    num_grupos <- length(unique(grupo))
    
    if (oper == "t") {
      if (num_grupos != 2 || length(variables) != 1) {
        output$resultado <- renderText("La prueba T requiere 2 grupos y solo una variable numérica.")
        output$grafico <- renderPlot(NULL)
        return()
      }
      
      y <- df[[variables[1]]]
      prueba <- t.test(y ~ grupo)
      stat <- prueba$statistic
      dfree <- prueba$parameter
      alpha <- 0.05
      x <- seq(-4, 4, length.out = 1000)
      crit <- qt(1 - alpha / 2, dfree)
      
      output$resultado <- renderPrint(prueba)
      
      output$grafico <- renderPlot({
        ggplot(data.frame(x = x), aes(x)) +
          stat_function(fun = dt, args = list(df = dfree), color = "blue") +
          geom_area(data = subset(data.frame(x), x <= -crit), aes(y = dt(x, dfree)), fill = "red", alpha = 0.3) +
          geom_area(data = subset(data.frame(x), x >= crit), aes(y = dt(x, dfree)), fill = "red", alpha = 0.3) +
          geom_vline(xintercept = stat, color = "darkgreen", size = 1.2) +
          labs(title = "Distribución t con áreas de rechazo",
               x = "t", y = "Densidad") +
          theme_minimal()
      })
      
    } else if (oper == "anova") {
      if (num_grupos < 2) {
        output$resultado <- renderText("ANOVA requiere al menos 3 grupos.")
        output$grafico <- renderPlot(NULL)
        return()
      }
      
      var <- variables[1]  # Solo graficamos la primera variable como ejemplo
      formula <- as.formula(paste(var, "~ grupo"))
      modelo <- aov(formula, data = df)
      re<-report(modelo)
      
      sum_mod <- summary(modelo)[[1]]
      stat <- sum_mod$Fvalue[1]
      
      df1 <- sum_mod$Df[1]
      df2 <- sum_mod$Df[2]
      
      alpha <- 0.05
      x <- seq(0, 6, length.out = 1000)
      crit <- qf(1 - alpha, df1, df2)
      
      resultados <- lapply(variables, function(v) {
        summary(aov(as.formula(paste(v, "~ grupo")), data = df))
      })
      names(resultados) <- variables
      
      output$resultado <- renderPrint(resultados)
      output$resumen_modelo <- renderPrint(sum_mod)
      output$r<-renderPrint(re)
      output$grafico <- renderPlot({
        ggplot(data.frame(x = x), aes(x)) +
          stat_function(fun = df, args = list(df1 = df1, df2 = df2), color = "blue") +
          geom_area(data = subset(data.frame(x), x >= crit), aes(y = df(x, df1, df2)), fill = "red", alpha = 0.3) +
          geom_vline(xintercept = stat, color = "darkgreen", size = 1.2) +
          labs(title = paste("Distribución F con área de rechazo -", var),
               x = "F", y = "Densidad") +
          theme_minimal()
        
      })
    }
  })
}

shinyApp(ui, server)