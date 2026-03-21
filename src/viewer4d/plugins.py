"""
Plugin loader for viewer4d.

Plugins live in the ``viewer4d_plugin`` namespace package.  Each sub-package
should expose a class with a ``plugin_id`` class attribute.  The class is
instantiated with ``PluginClass(parent=self, plugin_menu=..., plugin_action=...)``.

Plugin API
----------
Required:
  plugin_id     : str  — unique dotted identifier

Optional flags:
  display_name      : str   — human-readable name (used in menu)
  uses_plugin_menu  : bool  — if True, a QMenu sub-menu is created and passed
  uses_single_action: bool  — if True, a single QAction is created and passed

Example minimal plugin::

    class MyPlugin:
        plugin_id    = "me.my_plugin"
        display_name = "My Feature"
        uses_single_action = True

        def __init__(self, parent, plugin_action, **kwargs):
            plugin_action.triggered.connect(self.run)

        def run(self):
            ...

        def close(self):
            pass
"""

import pkgutil
import importlib
import inspect
import traceback

from PyQt5.QtWidgets import QMenu, QAction

__all__ = ["load_plugins"]


def load_plugins(self):
    try:
        import viewer4d_plugin
    except ImportError:
        print("viewer4d_plugin namespace not found; no plugins loaded.")
        return

    self.loaded_plugins = []

    for module_info in pkgutil.iter_modules(getattr(viewer4d_plugin, "__path__")):
        try:
            module = importlib.import_module(
                viewer4d_plugin.__name__ + "." + module_info.name
            )
        except Exception as e:
            print(f"Plugin import failed [{module_info.name}]: {e}")
            traceback.print_exc()
            continue

        for name, member in inspect.getmembers(module, inspect.isclass):
            plugin_id = getattr(member, "plugin_id", None)
            if not plugin_id:
                continue

            print(f"  Loading plugin: {plugin_id}  ({name})")
            try:
                plugin_menu = None
                if getattr(member, "uses_plugin_menu", False):
                    plugin_menu = QMenu(
                        getattr(member, "display_name", "Plugin"), self
                    )
                    self.processing_menu.addMenu(plugin_menu)

                plugin_action = None
                if getattr(member, "uses_single_action", False):
                    plugin_action = QAction(
                        getattr(member, "display_name", "Plugin"), self
                    )
                    self.processing_menu.addAction(plugin_action)

                instance = member(
                    parent=self,
                    plugin_menu=plugin_menu,
                    plugin_action=plugin_action,
                )
                self.loaded_plugins.append(
                    {"plugin": instance, "menu": plugin_menu, "action": plugin_action}
                )
            except Exception as exc:
                print(f"  Plugin init failed [{plugin_id}]: {exc}")
                traceback.print_exc()
