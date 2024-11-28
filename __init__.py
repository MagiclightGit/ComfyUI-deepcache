try:
	import comfy.utils
except ImportError:
	pass
else:
	from .DeepCache import DeepCache
	from .DeepCacheV2 import DeepCacheV2


NODE_CLASS_MAPPINGS = {
	"DeepCache": DeepCache,
	"DeepCacheV2": DeepCacheV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"DeepCache": "Deep Cache",
	"DeepCacheV2": "Deep Cache - V2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
