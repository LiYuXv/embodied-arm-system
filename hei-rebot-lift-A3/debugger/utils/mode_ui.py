"""Visibility helpers for existing per-arm layouts (no control callbacks)."""


def set_layout_visible(layout, visible):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is not None:
            item.widget().setVisible(visible)
        elif item.layout() is not None:
            set_layout_visible(item.layout(), visible)


def set_arm_selector(combo, dual):
    combo.setCurrentIndex(0)
    # Keep item data and translation indexes stable across mode changes.
    combo.model().item(1).setEnabled(dual)
    combo.view().setRowHidden(1, not dual)
    combo.setVisible(dual)
