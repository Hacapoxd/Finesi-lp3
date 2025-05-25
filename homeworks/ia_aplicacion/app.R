library(shiny)
library(pdftools)
library(stringr)
library(httr)
library(jsonlite)
library(shiny)
library(shiny)
library(bibliometrix)
library(ggplot2)
library(plotly)
library(DT)
library(igraph)
library(networkD3)
library(wordcloud)
library(tm)
library(dplyr)
library(topicmodels)
library(shinythemes)

extraer_secciones <- function(texto) {
  texto_lower <- tolower(texto)
  abstract_start <- str_locate(texto_lower, "abstract")[1,1]
  methods_start <- str_locate(texto_lower, "method")[1,1]
  results_start <- str_locate(texto_lower, "result")[1,1]
  conclusion_start <- str_locate(texto_lower, "conclusion")[1,1]
  
  secciones <- list()
  
  if (!is.na(abstract_start)) {
    end_abstract <- ifelse(!is.na(methods_start), methods_start - 1, nchar(texto))
    secciones$abstract <- str_sub(texto, abstract_start, end_abstract)
  }
  if (!is.na(methods_start)) {
    end_methods <- ifelse(!is.na(results_start), results_start - 1, nchar(texto))
    secciones$methods <- str_sub(texto, methods_start, end_methods)
  }
  if (!is.na(results_start)) {
    end_results <- ifelse(!is.na(conclusion_start), conclusion_start - 1, nchar(texto))
    secciones$results <- str_sub(texto, results_start, end_results)
  }
  if (!is.na(conclusion_start)) {
    secciones$conclusion <- str_sub(texto, conclusion_start, nchar(texto))
  }
  
  paste(unlist(secciones), collapse = "\n\n")
}

extraer_texto_ordenado <- function(pdf_path) {
  paginas <- pdf_data(pdf_path)
  
  texto_completo <- sapply(paginas, function(pg) {
    ancho_pag <- max(pg$x)
    mitad_x <- ancho_pag / 2
    
    # Separar en columnas izquierda y derecha
    columna_izquierda <- pg %>% filter(x <= mitad_x)
    columna_derecha  <- pg %>% filter(x > mitad_x)
    
    ordenar_columna <- function(columna) {
      columna %>%
        arrange(y, x) %>%
        mutate(linea = cumsum(c(1, diff(y) > 2))) %>%
        group_by(linea) %>%
        summarise(linea_texto = str_c(text, collapse = " ")) %>%
        pull(linea_texto)
    }
    
    texto_izq <- ordenar_columna(columna_izquierda)
    texto_der <- ordenar_columna(columna_derecha)
    
    # Unimos columnas en orden: primero izq, luego der (puedes invertir si quieres)
    paste(c(texto_izq, texto_der), collapse = "\n")
  })
  
  paste(texto_completo, collapse = "\n\n")
}



contar_tokens <- function(texto) {
  nchar(texto) / 4
}

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Análisis bibliométrico con GPT - Estadística Computacional"),
  sidebarLayout(
    
    sidebarPanel(
      
      fileInput("pdf_file", "Sube un archivo PDF", accept = ".pdf"),
      actionButton("analyze_btn", "Extraer texto y calcular costo"),
      checkboxInput("extraer_avanzado", "Usar extracción avanzada para PDF con columnas/doble cara", value = TRUE),
      hr()
      
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Texto y Análisis GPT",
                 h4("texto_analisis_gpt"),
                 verbatimTextOutput("token_info"),
                 actionButton("confirm_gpt", "Confirmar análisis con GPT"),
                 
                 h4("Texto extraído para análisis"),
                 div(
                   style = "max-height: 400px; overflow-y: auto; background-color: #f8f9fa; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap;",
                   verbatimTextOutput("extracted_text")
                 ),
                 hr(),
                 h4("Respuesta de GPT"),
                 verbatimTextOutput("gpt_response"),
                 
        ),
        tabPanel("Bibliometría Básica",
                 h4("Análisis Bibliométrico GPT"),
                 verbatimTextOutput("biblio_token_info"),
                 actionButton("auto_btn", "Análisis Automático (Bibliometría)"),
                 
                 h4("Citas encontradas en el texto"),
                 DTOutput("tabla_citas"),
                 
                 h4("Bibliometría por GPT"),
                 verbatimTextOutput("biblio_gpt_response"),
                 
        ),
        tabPanel("Referencias Detectadas",
                 DTOutput("tabla_referencias")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  extraer_referencias <- function(texto) {
    seccion_ref <- max(unlist(gregexpr("(?i)\\b(referencias|bibliografía|references)\\b", texto)))
    
    if (is.na(seccion_ref)) return(character(0))
    
    refs_texto <- substr(texto, seccion_ref, nchar(texto))
    
    # Mejor división de referencias
    posibles_refs <- unlist(str_split(refs_texto, "\n\\d+\\.|\\n\\[[0-9]+\\]|\\n[A-Z][a-z]+,"))
    
    posibles_refs <- posibles_refs[nchar(posibles_refs) > 30]
    posibles_refs <- trimws(posibles_refs)
    
    return(posibles_refs)
  }
  
  extraer_citas <- function(texto) {
    patron <- "\\b([A-Z][a-z]+(?:\\set\\sal\\.)?,?\\s?(?:[A-Z]\\.)?\\s?\\(?\\d{4}\\)?)"
    citas <- str_extract_all(texto, patron)[[1]]
    citas <- citas[!is.na(citas)]
    citas <- trimws(citas)
    citas
  }
  
  procesar_citas <- function(citas) {
    df <- data.frame(Cita = citas, stringsAsFactors = FALSE)
    df <- df %>%
      mutate(Autor = str_extract(Cita, "^[A-Z][a-z]+"),
             Anio = str_extract(Cita, "\\d{4}")) %>%
      filter(!is.na(Autor) & !is.na(Anio))
    df
  }
  
  referencias_detectadas <- reactive({
    req(texto_extraido())
    extraer_referencias(texto_extraido())
  })
  
  output$tabla_referencias <- renderDT({
    data.frame(Referencia = referencias_detectadas(), stringsAsFactors = FALSE)
  })
  
  
  texto_extraido <- reactiveVal("")
  
  observeEvent(input$analyze_btn, {
    req(input$pdf_file)
    if (isTRUE(input$extraer_avanzado)) {
      texto_completo <- extraer_texto_ordenado(input$pdf_file$datapath)
    } else {
      texto_pdf <- pdf_text(input$pdf_file$datapath)
      texto_completo <- paste(texto_pdf, collapse = "\n\n")
    }
    
    texto_clave <- extraer_secciones(texto_completo)
    if (texto_clave == "") {
      texto_clave <- substr(texto_completo, 1, 4000)
    }
    
    texto_extraido(texto_clave)
  })
  
  
  
  output$extracted_text <- renderText({
    texto_extraido()
  })
  
  citas_df <- reactive({
    req(texto_extraido())
    citas <- extraer_citas(texto_extraido())
    procesar_citas(citas)
  })
  
  output$tabla_citas <- renderDT({
    citas_df()
  })
  
  output$wordcloud_autores <- renderPlot({
    autores <- citas_df()$Autor
    if (length(autores) > 0) {
      wordcloud(autores, max.words = 50, colors = brewer.pal(8, "Dark2"))
    }
  })
  
  output$grafico_anios <- renderPlot({
    datos <- citas_df() %>%
      count(Anio)
    
    ggplot(datos, aes(x = Anio, y = n)) +
      geom_col(fill = "steelblue") +
      theme_minimal() +
      labs(title = "Citas por Año", x = "Año", y = "Cantidad")
  })
  
  
  output$token_info <- renderText({
    req(texto_extraido())
    tokens <- round(contar_tokens(texto_extraido()))
    costo_aprox <- tokens * 0.00003
    paste0("Tokens aproximados a enviar: ", tokens,
           "\nCosto estimado de la consulta: $", round(costo_aprox, 4))
  })
  output$biblio_token_info <- renderText({
    req(texto_extraido())
    tokens <- round(contar_tokens(texto_extraido()))
    costo_aprox <- tokens * 0.00003
    paste0("Tokens usados para bibliometría: ", tokens,
           "\nCosto estimado de la consulta: $", round(costo_aprox, 4))
  })
  
  call_openai_api <- function(prompt_text) {
    api_key <- Sys.getenv("OPENAI_API_KEY")
    if (api_key == "") return("No API key set.")
    
    url <- "https://api.openai.com/v1/chat/completions"
    body <- list(
      model = "gpt-4o-mini",
      messages = list(
        list(role = "system", content = "You are an expert in computational statistics. Analyze the text and extract key info: Study, Type, Dataset, Techniques, Association, Impact, DFD, Summary."),
        list(role = "user", content = prompt_text)
      ),
      max_tokens = 500,
      temperature = 0.3
    )
    
    res <- POST(url, add_headers(Authorization = paste("Bearer", api_key)),
                body = toJSON(body, auto_unbox = TRUE), encode = "json")
    
    if (res$status_code != 200) {
      return(paste("Error:", res$status_code, content(res)$error$message))
    }
    
    content(res)$choices[[1]]$message$content
  }
  call_biblio_api <- function(prompt_text) {
    api_key <- Sys.getenv("OPENAI_API_KEY")
    if (api_key == "") return("No API key set.")
    
    url <- "https://api.openai.com/v1/chat/completions"
    body <- list(
      model = "gpt-4o-mini",
      messages = list(
        list(role = "system", content = "You are a bibliometrics expert. Detect patterns, key authors, countries, and trends in scientific production."),
        list(role = "user", content = prompt_text)
      ),
      max_tokens = 500,
      temperature = 0.4
    )
    
    res <- POST(url, add_headers(Authorization = paste("Bearer", api_key)),
                body = toJSON(body, auto_unbox = TRUE), encode = "json")
    
    if (res$status_code != 200) {
      return(paste("Error:", res$status_code, content(res)$error$message))
    }
    
    content(res)$choices[[1]]$message$content
  }
  
  
  gpt_result <- eventReactive(input$confirm_gpt, {
    req(texto_extraido())
    call_openai_api(texto_extraido())
  })
  biblio_gpt_result <- eventReactive(input$auto_btn, {
    req(texto_extraido())
    call_biblio_api(texto_extraido())
  })
  
  
  output$gpt_response <- renderText({
    gpt_result()
  })
  output$biblio_gpt_response <- renderText({
    biblio_gpt_result()
  })
  
  
  
}

shinyApp(ui, server)