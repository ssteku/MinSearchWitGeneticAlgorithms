import pygal                                                       # First import pygal

class ChartDrawer:
    def draw(self, name, seriesName, results, xLabels, xTitle, yTitle):
        self.line_chart = pygal.Line()
        self.line_chart.title = name
        self.line_chart.x_labels = xLabels
        self.line_chart.x_title = xTitle
        self.line_chart.y_title = yTitle
        self.line_chart.add(seriesName, results)
        self.line_chart.render_to_file("Charts/"+seriesName) 

