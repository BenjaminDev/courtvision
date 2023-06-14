from textual.app import App, ComposeResult
from textual.widgets import DirectoryTree

# class DirectoryTreeApp(App):
#     def compose(self) -> ComposeResult:
#         yield DirectoryTree("./")


# if __name__ == "__main__":
#     app = DirectoryTreeApp()
#     app.run()


def review():
    """
    Review the latest match in the database.
    """
    from courtvision.models import get_fasterrcnn_ball_detection_model
