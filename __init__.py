"""Hermes model-provider discovery entrypoint (not a generic tool plugin).

Hermes loads this file from ``$HERMES_HOME/plugins/model-providers/melious/`` and mirrors
the registered profile into its provider registry. See ``melious_provider.py`` for the
behaviour and ``README.md`` for the install/verify/uninstall flow.
"""

from providers import register_provider

from .melious_provider import profile

register_provider(profile)
