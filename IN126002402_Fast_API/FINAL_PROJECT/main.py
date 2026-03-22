from fastapi import FastAPI, Query, Response
from pydantic import BaseModel, Field
from typing import Optional
import math

app = FastAPI()

menu = [
    {"id": 1, "name": "Margherita Pizza", "price": 250, "category": "Pizza", "is_available": True},
    {"id": 2, "name": "Veg Burger", "price": 120, "category": "Burger", "is_available": True},
    {"id": 3, "name": "Coke", "price": 50, "category": "Drink", "is_available": True},
    {"id": 4, "name": "Chocolate Cake", "price": 180, "category": "Dessert", "is_available": False},
    {"id": 5, "name": "Pepperoni Pizza", "price": 300, "category": "Pizza", "is_available": True},
    {"id": 6, "name": "French Fries", "price": 100, "category": "Snack", "is_available": True}
]

orders = []
order_counter = 1

cart = []


class OrderRequest(BaseModel):
    customer_name: str = Field(min_length=2)
    item_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=20)
    delivery_address: str = Field(min_length=10)
    order_type: str = "delivery"


class NewMenuItem(BaseModel):
    name: str = Field(min_length=2)
    price: int = Field(gt=0)
    category: str = Field(min_length=2)
    is_available: bool = True


class CheckoutRequest(BaseModel):
    customer_name: str
    delivery_address: str


def find_menu_item(item_id):
    for item in menu:
        if item["id"] == item_id:
            return item
    return None


def calculate_bill(price, quantity, order_type="delivery"):
    total = price * quantity
    if order_type == "delivery":
        total += 30
    return total


def filter_menu_logic(category, max_price, is_available):
    result = menu
    if category is not None:
        result = [i for i in result if i["category"].lower() == category.lower()]
    if max_price is not None:
        result = [i for i in result if i["price"] <= max_price]
    if is_available is not None:
        result = [i for i in result if i["is_available"] == is_available]
    return result


@app.get("/")
def home():
    return {"message": "Welcome to QuickBite Food Delivery"}


@app.get("/menu")
def get_menu():
    return {"total": len(menu), "items": menu}


@app.get("/menu/summary")
def menu_summary():
    available = len([i for i in menu if i["is_available"]])
    unavailable = len(menu) - available
    categories = list(set([i["category"] for i in menu]))
    return {
        "total": len(menu),
        "available": available,
        "unavailable": unavailable,
        "categories": categories
    }


@app.get("/menu/filter")
def filter_menu(category: Optional[str] = None, max_price: Optional[int] = None, is_available: Optional[bool] = None):
    result = filter_menu_logic(category, max_price, is_available)
    return {"count": len(result), "items": result}


@app.get("/menu/search")
def search_menu(keyword: str):
    result = [i for i in menu if keyword.lower() in i["name"].lower() or keyword.lower() in i["category"].lower()]
    if not result:
        return {"message": "No items found"}
    return {"total_found": len(result), "items": result}


@app.get("/menu/sort")
def sort_menu(sort_by: str = "price", order: str = "asc"):
    if sort_by not in ["price", "name", "category"]:
        return {"error": "Invalid sort_by"}
    if order not in ["asc", "desc"]:
        return {"error": "Invalid order"}
    reverse = True if order == "desc" else False
    sorted_menu = sorted(menu, key=lambda x: x[sort_by], reverse=reverse)
    return {"sorted_by": sort_by, "order": order, "items": sorted_menu}


@app.get("/menu/page")
def paginate_menu(page: int = Query(1, ge=1), limit: int = Query(3, ge=1, le=10)):
    start = (page - 1) * limit
    data = menu[start:start + limit]
    total_pages = math.ceil(len(menu) / limit)
    return {
        "page": page,
        "limit": limit,
        "total": len(menu),
        "total_pages": total_pages,
        "items": data
    }


@app.get("/menu/browse")
def browse_menu(keyword: Optional[str] = None, sort_by: str = "price", order: str = "asc", page: int = 1, limit: int = 4):
    result = menu
    if keyword:
        result = [i for i in result if keyword.lower() in i["name"].lower() or keyword.lower() in i["category"].lower()]
    reverse = True if order == "desc" else False
    result = sorted(result, key=lambda x: x[sort_by], reverse=reverse)
    start = (page - 1) * limit
    data = result[start:start + limit]
    total_pages = math.ceil(len(result) / limit)
    return {
        "total": len(result),
        "page": page,
        "total_pages": total_pages,
        "items": data
    }


@app.get("/menu/{item_id}")
def get_item(item_id: int):
    item = find_menu_item(item_id)
    if not item:
        return {"error": "Item not found"}
    return item


@app.post("/menu")
def add_menu(item: NewMenuItem, response: Response):
    for i in menu:
        if i["name"].lower() == item.name.lower():
            return {"error": "Item already exists"}
    new_item = item.dict()
    new_item["id"] = len(menu) + 1
    menu.append(new_item)
    response.status_code = 201
    return new_item


@app.put("/menu/{item_id}")
def update_menu(item_id: int, price: Optional[int] = None, is_available: Optional[bool] = None):
    item = find_menu_item(item_id)
    if not item:
        return {"error": "Item not found"}
    if price is not None:
        item["price"] = price
    if is_available is not None:
        item["is_available"] = is_available
    return item


@app.delete("/menu/{item_id}")
def delete_menu(item_id: int):
    item = find_menu_item(item_id)
    if not item:
        return {"error": "Item not found"}
    menu.remove(item)
    return {"message": "Deleted", "item": item["name"]}


@app.get("/orders")
def get_orders():
    return {"total_orders": len(orders), "orders": orders}


@app.get("/orders/search")
def search_orders(customer_name: str):
    result = [o for o in orders if customer_name.lower() in o["customer_name"].lower()]
    return {"count": len(result), "orders": result}


@app.get("/orders/sort")
def sort_orders(order: str = "asc"):
    reverse = True if order == "desc" else False
    sorted_orders = sorted(orders, key=lambda x: x["total_price"], reverse=reverse)
    return {"orders": sorted_orders}


@app.post("/orders")
def create_order(order: OrderRequest):
    global order_counter
    item = find_menu_item(order.item_id)
    if not item:
        return {"error": "Item not found"}
    if not item["is_available"]:
        return {"error": "Item not available"}
    total = calculate_bill(item["price"], order.quantity, order.order_type)
    new_order = {
        "order_id": order_counter,
        "customer_name": order.customer_name,
        "item": item["name"],
        "quantity": order.quantity,
        "total_price": total,
        "address": order.delivery_address
    }
    orders.append(new_order)
    order_counter += 1
    return new_order


@app.post("/cart/add")
def add_to_cart(item_id: int, quantity: int = 1):
    item = find_menu_item(item_id)
    if not item:
        return {"error": "Item not found"}
    if not item["is_available"]:
        return {"error": "Not available"}
    for c in cart:
        if c["item_id"] == item_id:
            c["quantity"] += quantity
            return {"message": "Updated cart", "cart": cart}
    cart.append({"item_id": item_id, "name": item["name"], "price": item["price"], "quantity": quantity})
    return {"message": "Added to cart", "cart": cart}


@app.get("/cart")
def get_cart():
    total = sum([i["price"] * i["quantity"] for i in cart])
    return {"cart": cart, "grand_total": total}


@app.delete("/cart/{item_id}")
def remove_cart(item_id: int):
    for i in cart:
        if i["item_id"] == item_id:
            cart.remove(i)
            return {"message": "Removed"}
    return {"error": "Item not in cart"}


@app.post("/cart/checkout")
def checkout(data: CheckoutRequest, response: Response):
    global order_counter
    if not cart:
        return {"error": "Cart is empty"}
    new_orders = []
    total = 0
    for c in cart:
        item = find_menu_item(c["item_id"])
        bill = calculate_bill(item["price"], c["quantity"])
        order = {
            "order_id": order_counter,
            "customer_name": data.customer_name,
            "item": item["name"],
            "quantity": c["quantity"],
            "total_price": bill,
            "address": data.delivery_address
        }
        orders.append(order)
        new_orders.append(order)
        total += bill
        order_counter += 1
    cart.clear()
    response.status_code = 201
    return {"orders": new_orders, "total": total}