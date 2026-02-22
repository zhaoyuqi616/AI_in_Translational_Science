library(shiny)
library(bslib)
library(dplyr)
library(ggplot2)
library(tidyr)
library(plotly)
library(heatmaply)
library(DT)
library(thematic)
library(bsicons)

# Load data
# Ensure ccle_tiny.rds is in the same folder as app.R
full_data <- readRDS("ccle_tiny.rds")
lineages <- sort(unique(full_data$lineage))

ui <- navbarPage(
  title = "🧬 CCLE Explorer Pro",
  theme = bslib::bs_theme(
    bootswatch = "flatly", 
    base_font = "Inter, sans-serif" 
  ),
  
  # Dark Mode Toggle
  footer = div(
    style = "position: fixed; bottom: 20px; right: 20px; z-index: 1000;",
    input_dark_mode(id = "dark_mode")
  ),

  tabPanel("Single Gene Explorer",
    layout_sidebar(
      sidebar = sidebar(
        title = "Parameters",
        selectizeInput("gene_search", "Search Gene Symbol:", choices = NULL, selected = "TP53"),
        selectInput("plot_type", "Plot Style:", choices = c("Bar Plot" = "bar", "Lollipop" = "dash")),
        sliderInput("num_cells", "Top N Cell Lines:", min = 5, max = 100, value = 30)
      ),
      card(
        card_header("Gene Expression Levels"),
        plotlyOutput("genePlot", height = "500px")
      ),
      card(
        card_header("Raw Data Table"),
        DT::DTOutput("dataTable")
      )
    )
  ),

  tabPanel("Gene Correlation",
    layout_sidebar(
      sidebar = sidebar(
        selectizeInput("gene_x", "Gene X:", choices = NULL, selected = "KRAS"),
        selectizeInput("gene_y", "Gene Y:", choices = NULL, selected = "BRAF")
      ),
      card(
        card_header("Co-expression Analysis"),
        plotlyOutput("corrPlot", height = "600px")
      )
    )
  ),

  tabPanel("Differential Expression",
    layout_sidebar(
      sidebar = sidebar(
        selectizeInput("de_gene", "Select Gene:", choices = NULL),
        selectInput("group_a", "Group A:", choices = lineages, selected = "Lung"),
        selectInput("group_b", "Group B:", choices = lineages, selected = "Skin")
      ),
      layout_column_wrap(
        width = 1,
        value_box(
          title = "Statistical Significance",
          value = textOutput("p_val_text"),
          showcase = bsicons::bs_icon("graph-up"),
          theme = "primary"
        ),
        card(
          card_header("Lineage Comparison"),
          plotlyOutput("de_box_plot")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  # This matches plot colors to Dark/Light mode automatically
  thematic::thematic_shiny()

  # Server-side selectize for speed
  gene_list <- sort(unique(full_data$gene_name))
  updateSelectizeInput(session, "gene_search", choices = gene_list, server = TRUE)
  updateSelectizeInput(session, "gene_x", choices = gene_list, server = TRUE)
  updateSelectizeInput(session, "gene_y", choices = gene_list, server = TRUE)
  updateSelectizeInput(session, "de_gene", choices = gene_list, server = TRUE)

  # Logic for Tab 1
  gene_data <- reactive({
    req(input$gene_search)
    full_data %>% 
      filter(gene_name == input$gene_search) %>%
      arrange(desc(rna_expression)) %>% 
      slice_head(n = input$num_cells)
  })

  output$genePlot <- renderPlotly({
    p <- ggplot(gene_data(), aes(x = reorder(cell_line, rna_expression), y = rna_expression, fill = lineage)) +
      geom_bar(stat="identity") +
      theme_minimal() + 
      labs(x = "Cell Line", y = "log2(TPM+1)") +
      theme(axis.text.x = element_blank())
    ggplotly(p)
  })

  output$dataTable <- DT::renderDT({
    gene_data()
  })

  # Logic for Tab 2
  output$corrPlot <- renderPlotly({
    req(input$gene_x, input$gene_y)
    corr_df <- full_data %>%
      filter(gene_name %in% c(input$gene_x, input$gene_y)) %>%
      pivot_wider(names_from = gene_name, values_from = rna_expression)
    
    p <- ggplot(corr_df, aes_string(x = paste0("`", input$gene_x, "`"), y = paste0("`", input$gene_y, "`"), color = "lineage")) +
      geom_point(alpha = 0.6) +
      theme_minimal()
    ggplotly(p)
  })

  # Logic for Tab 3
  output$p_val_text <- renderText({
    req(input$de_gene)
    df <- full_data %>% filter(gene_name == input$de_gene, lineage %in% c(input$group_a, input$group_b))
    if(length(unique(df$lineage)) < 2) return("N/A")
    t_test <- t.test(rna_expression ~ lineage, data = df)
    paste("P-value:", format.pval(t_test$p.value, digits = 3))
  })

  output$de_box_plot <- renderPlotly({
    req(input$de_gene)
    df <- full_data %>% filter(gene_name == input$de_gene, lineage %in% c(input$group_a, input$group_b))
    p <- ggplot(df, aes(x = lineage, y = rna_expression, fill = lineage)) +
      geom_boxplot() +
      theme_minimal()
    ggplotly(p)
  })
}

shinyApp(ui, server)