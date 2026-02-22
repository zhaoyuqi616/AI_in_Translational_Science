library(shiny)
library(bslib)
library(DT)

full_data <- readRDS("ccle_tiny.rds")
gene_list <- sort(unique(full_data$gene_name))

ui <- navbarPage(
  title = "🧬 CCLE Explorer",
  theme = bs_theme(bootswatch = "flatly"),
  
  # Tab 1: Single Gene
  tabPanel("Single Gene",
    layout_sidebar(
      sidebar = sidebar(
        selectInput("gene", "Search Gene:", choices = gene_list, selected = "TP53"),
        sliderInput("num_cells", "Show Top N:", min = 5, max = 30, value = 15)
      ),
      card(card_header("Expression Profile"), plotOutput("genePlot", height = "500px"))
    )
  ),
  
  # Tab 2: Gene Correlation
  tabPanel("Gene Correlation",
    layout_sidebar(
      sidebar = sidebar(
        selectInput("gene_x", "Gene X:", choices = gene_list, selected = "KRAS"),
        selectInput("gene_y", "Gene Y:", choices = gene_list, selected = "BRAF")
      ),
      card(card_header("Co-expression Analysis"), plotOutput("corrPlot", height = "500px"))
    )
  ),

  # Tab 3: Data Explorer with Download
  tabPanel("Data Explorer",
    layout_sidebar(
      sidebar = sidebar(
        downloadButton("downloadData", "Download CSV", class = "btn-primary")
      ),
      card(
        card_header("Filter and Search Raw Data"),
        DT::DTOutput("rawTable")
      )
    )
  )
)

server <- function(input, output) {
  
  # Logic for Tab 1
  output$genePlot <- renderPlot({
    req(input$gene)
    df <- full_data[full_data$gene_name == input$gene, ]
    df <- df[order(-df$rna_expression), ]
    plot_df <- head(df, input$num_cells)
    par(mar = c(10, 5, 4, 2)) 
    b <- barplot(plot_df$rna_expression, col = "#2c3e50", border = "white",
            main = paste("Top Cell Lines for", input$gene), ylab = "log2(TPM+1)")
    text(x = b, y = -0.2, labels = plot_df$cell_line, srt = 45, adj = 1, xpd = TRUE, cex = 0.8)
  })

  # Logic for Tab 2
  output$corrPlot <- renderPlot({
    req(input$gene_x, input$gene_y)
    data_x <- full_data[full_data$gene_name == input$gene_x, c("cell_line", "rna_expression")]
    data_y <- full_data[full_data$gene_name == input$gene_y, c("cell_line", "rna_expression")]
    merged <- merge(data_x, data_y, by = "cell_line")
    plot(merged$rna_expression.x, merged$rna_expression.y,
         xlab = input$gene_x, ylab = input$gene_y, pch = 19, col = rgb(0.17, 0.24, 0.31, 0.5))
    abline(lm(rna_expression.y ~ rna_expression.x, data = merged), col = "red", lwd = 2)
  })

  # Logic for Tab 3
  output$rawTable <- DT::renderDT({
    DT::datatable(full_data, filter = "top", options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
  })

  # --- NEW: Download Logic ---
  output$downloadData <- downloadHandler(
    filename = function() { paste("ccle_data_", Sys.Date(), ".csv", sep = "") },
    content = function(file) { write.csv(full_data, file, row.names = FALSE) }
  )
}

shinyApp(ui, server)
