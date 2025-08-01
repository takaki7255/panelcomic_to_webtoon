import xml.etree.ElementTree as ET


def get_framebbox(xml_file: str) -> list:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    pages = root.findall('.//page')
    page_objects = {}
    for page in pages:
        page_index = page.get('index')
        page_index = int(page_index)
        objects = []
        for obj in page:
            if obj.tag in ["frame"]:
                obj_data = {
                    "type": obj.tag,
                    "id": obj.get("id"),
                    "xmin": obj.get("xmin"),
                    "ymin": obj.get("ymin"),
                    "xmax": obj.get("xmax"),
                    "ymax": obj.get("ymax"),
                }
                objects.append(obj_data)
        page_objects[page_index] = objects
    return page_objects