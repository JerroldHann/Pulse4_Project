def ask_agent(query: str) -> str:
    if "client" in query.lower():
        return "Client A shows medium risk: unusual transaction frequency with 3 connected accounts."
    elif "transaction" in query.lower():
        return "Transaction 2024-0923 is flagged as high-risk due to amount anomaly."
    else:
        return "Please specify a client or transaction for analysis."
