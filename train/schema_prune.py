import json
from collections import Counter

from filter_utils import extract_schema_json

# filtered_data_p = "train/data/chat_templated_jsonschema_dataset_max1024.json"
valid_data_p = "train/data/valid.json"
# with open(filtered_data_p, "r", encoding="utf-8") as f:
#     data = json.load(f)

# data = [extract_schema_json(elem['text']) for elem in data if extract_schema_json(elem['text'])]
# with open(valid_data_p, "w", encoding="utf-8") as f:
#     json.dump(data, f)

freq_keys = [('$schema', 4270), ('type', 3832), ('properties', 3647), ('title', 2640), ('required', 2586), ('description', 1590), ('$id', 1411), ('additionalProperties', 1148), ('definitions', 901), ('id', 326), ('$ref', 261), ('examples', 201), ('$defs', 196), ('default', 175), ('items', 164)]
trivial_keys = [('javaType', 95), ('javaInterfaces', 79), ('allOf', 72), ('version', 58), ('cli', 55), ('patternProperties', 39), ('oneOf', 35), ('$comment', 34), ('additonalProperties', 30), ('anyOf', 28), ('meta', 28), ('allowComments', 23), ('dependencies', 21), ('propertyNames', 19), ('if', 16), ('then', 16), ('additionalItems', 16), ('uniqueItems', 15), ('mapsToTagName', 14), ('inputType', 13), ('minItems', 11), ('minProperties', 10), ('self', 9), ('x-kubernetes-group-version-kind.group', 7), ('x-kubernetes-group-version-kind.kind', 7), ('x-kubernetes-group-version-kind.version', 7), ('else', 7), ('unevaluatedProperties', 7), ('name', 7), ('$metadata', 6), ('meta:extensible', 5), ('meta:status', 5), ('meta:xdmType', 5), ('meta:xdmId', 5), ('meta:altId', 5), ('outputType', 5), ('not', 5), ('$version', 5), ('field_order', 5), ('lastModified', 5), ('mozPipelineMetadata', 4), ('$async', 4), ('configFile', 4), ('maxProperties', 4), ('meta:abstract', 4), ('maxItems', 4), ('buttons', 4), ('fileMatch', 4), ('long-description', 4), ('$$description', 3), ('components', 3), ('links', 3), ('minLength', 3), ('identificador', 3), ('datos', 3), ('outputCapture', 3), ('errorMessage', 3), ('data', 2), ('connectorName', 2), ('interaction', 2), ('list', 2), ('meta:titleId', 2), ('meta:descriptionId', 2), ('subtopic', 2), ('metamodel_version', 2), ('optional', 2), ('$code', 2), ('url', 2), ('xUidPrefix', 2), ('descriptions', 1), ('widget', 1), ('mutable', 1), ('xrefProperties', 1), ('$docs', 1), ('propertyPattern', 1), ('@id', 1), ('nick', 1), ('extends', 1), ('category', 1), ('markdown', 1), ('core', 1), ('x-customProperty', 1), ('actual_parameters', 1), ('namespaces', 1), ('metadata', 1), ('defaultSnippets', 1), ('Filename', 1), ('AbsolutePath', 1), ('ArrayOfAbsolutePaths', 1), ('ArrayOfStrings', 1), ('AccountIdString', 1), ('NonEmptyString', 1), ('RegionString', 1), ('__HOW_TO_ADD_PROPERTY__', 1), ('__SCHEMA_DOCS__', 1), ('$anchor', 1), ('meta:createdDate', 1), ('invisible', 1), ('todo-array', 1), ('todo', 1), ('_controlsOrder', 1), ('_show_form_view', 1), ('$$target', 1), ('options', 1), ('translations', 1), ('gists', 1), ('identifier', 1), ('reqObj', 1), ('process', 1), ('schemas', 1), ('className', 1), ('comments', 1), ('block', 1), ('sortBy', 1), ('markdownDescription', 1), ('documentVersion', 1), ('annotations', 1), ('vc', 1), ('xdm:displayName', 1), ('xdm:profileImage', 1), ('xdm:profileLink', 1), ('@context', 1), ('syncthing', 1), ('hooks', 1), ('readOnly', 1), ('default percent to box', 1), ('descripten', 1), ('$exportedModuleInfo', 1), ('form', 1), ('x-vendia-indexes', 1), ('x-vendia-acls', 1), ('examplesFile', 1)]

with open(valid_data_p, "r") as jfd:
    data = json.load(jfd)

stat = {}
for tk, _ in trivial_keys:
    for schema in data:
        if tk in schema:
            if tk in stat:
                stat[tk].append(schema)
            else:
                stat[tk] = [schema]
print(stat)