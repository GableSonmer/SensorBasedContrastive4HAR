# # grid accuracy of different 1st transformation and 2nd transformation
# n = len(transformations)
# grid_linear = np.zeros((n, n))
# grid_fine = np.zeros((n, n))
#
# ####################### Grid Fine-tune Evaluation #######################
# for i in range(n):
#     for j in [4, 5, 6]:
#         trans1 = transformations[i]
#         trans2 = transformations[j]
#         acc1, acc2 = train_simclr(
#             x_data=data, y_data=label,
#             transform1=trans1,
#             transform2=trans2
#         )
#         grid_linear[i, j] = round(acc1, 2)
#         grid_fine[i, j] = round(acc2, 2)
#         print(
#             f'Pair {transformations_text[i]} & {transformations_text[j]}: Linear: {acc1:.2f} | Fine-tune: {acc2:.2f}')
# ####################### Grid Fine-tune Evaluation #######################
#
# # save grid learner to csv
# np.savetxt('linear.csv', grid_linear, fmt='%.4f', delimiter=',')
# np.savetxt('fine.csv', grid_fine, fmt='%.4f', delimiter=',')
#
# # use seaborn to plot the grid accuracy
# plt.figure(figsize=(30, 15), dpi=100)
#
# # 第一个子图
# plt.subplot(1, 2, 1)
# sns.heatmap(grid_linear, annot=True, cmap='YlGnBu', cbar=False,
#             xticklabels=transformations_text, yticklabels=transformations_text,
#             annot_kws={"size": 25})  # 调整注释文字大小
# plt.title('Linear Evaluation', fontsize=30)  # 调整标题大小
# plt.xticks(rotation=90, fontsize=30)
# plt.yticks(fontsize=30)
#
# # 第二个子图
# plt.subplot(1, 2, 2)
# sns.heatmap(grid_fine, annot=True, cmap='YlGnBu', cbar=False, xticklabels=transformations_text,
#             annot_kws={"size": 25})  # 同样调整注释文字大小
# plt.title('Fine-tune Evaluation', fontsize=30)  # 调整标题大小
# plt.xticks(rotation=90, fontsize=30)
# plt.yticks(fontsize=30)
#
# plt.subplots_adjust(left=0.25, bottom=0.4, right=0.99, top=0.95)
#
# plt.savefig('contrastive_model/simclr/acc_heatmap.png')
# plt.show()
