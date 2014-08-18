import pygal                                                       # First import pygal
from pygal.style import CleanStyle
class ChartDrawer:
    def draw(self, name, seriesName, results, xLabels, xTitle, yTitle):
        self.line_chart = pygal.Line(
        	fill=True, style=CleanStyle,
        	x_label_rotation=45)
        self.line_chart.title = name
        self.line_chart.x_labels = xLabels
        self.line_chart.x_title = xTitle
        self.line_chart.y_title = yTitle
        self.line_chart.add(seriesName, results)
        self.line_chart.render_to_file("Charts/"+seriesName) 

