from generate import generate_reply

print("🌸 Akane is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("🌸 Akane: Bye! We Can Talk soon 💖")
        break

    reply = generate_reply(user_input, max_len=12, temperature=0.8, top_k=10)
    
    # Ensure short mini-chat reply
    if len(reply) > 50:
        # Take only first sentence or first 10 words
        reply = reply.split('.')[0] if '.' in reply else ' '.join(reply.split()[:10])
    
    # Ensure reply is not empty
    if not reply:
        reply = "mhmm~ 🌸"

    print("Akane:", reply)
